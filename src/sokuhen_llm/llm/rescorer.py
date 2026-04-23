"""LLM rescoring over a Viterbi ConversionResult.

The classical IME pipeline produces a ConversionResult with, per
segment, a small list of candidate surfaces ranked by unigram +
bigram cost. That ranking is usually right, but it gets confused by
longer-range context that only a real language model can resolve
("会議で事故が発生" vs "会議でじこが発生"; "使用" vs "しよう" mid-
sentence; etc.).

This rescorer takes the Viterbi output and, for each segment, tries
each candidate while holding the other choices fixed. The resulting
surface sequence is scored by the LLM; the highest-scoring candidate
is picked. A single pass is usually enough — correlations between
segments are handled implicitly because each segment's scoring sees
the earlier segments already re-decided.

Trade-offs
* O(sum(len(candidates_i))) LLM calls per rescoring run — for the
  typical Viterbi output of 5-8 segments with ~3 candidates each,
  that's ~20 scoring calls. At ~10 ms / call on CPU for the default
  110M-param model, one rescoring pass fits in ~200 ms — fast enough
  to run on commit (Enter) without user-visible delay, but too slow
  for every-keystroke live rescoring. We trigger it on commit only.
* The ``score_threshold`` gates re-ranking: if the LLM's preferred
  candidate beats the Viterbi winner by less than this log-prob
  delta, we stick with Viterbi. Prevents LLM from overriding strong
  dictionary evidence on close calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..engine.dictionary import DictEntry, Dictionary
from ..engine.viterbi import ConversionResult, ConversionSegment
from .backend import Backend, DummyBackend


log = logging.getLogger(__name__)


@dataclass
class RescoreConfig:
    """Knobs for the rescoring pass."""

    # Log-prob delta required for LLM to override the Viterbi choice.
    #
    # 1.5 nats = the LLM must be ~4.5× more confident in the alternative
    # before overriding the dictionary. This combines with the augmented
    # kana pool (``always_include_kana_forms``) to give the user's
    # requested "LLMが変換を決める" behavior without breaking style:
    #
    #   * Archaic-kanji rescue (縷→る, 鳴ん→なん): LLM's preference is
    #     many nats strong because the archaic form is nonsensical in
    #     context -- it easily clears 1.5 nats.
    #   * Wrong-kanji-among-commonly-used (事故 vs 自己, 使用 vs しよう):
    #     real-context disambiguations are 2+ nats -- clears 1.5.
    #   * Style-preference flips (及び↔および, 等↔など, 主に↔おもに):
    #     web-trained LM prefers hiragana by 0.3-1.0 nats -- fails 1.5,
    #     so dictionary's Wikipedia-formal kanji stays.
    #   * Noise flips (区画→区が, 呼称→湖沼, 機能→昨日, 以下→行か):
    #     wrong-kanji-but-similar-context -- fails 1.5.
    #
    # An earlier version of this config set threshold=0.0 to honor the
    # user's "LLMが決める" request literally, but that allowed all of
    # the above failure modes in, dropping wiki 89→81. The real way to
    # honor the request is: make the kana form AVAILABLE to the LLM
    # (via augmentation), but gate flips behind a "decisive confidence"
    # threshold so noise-level preferences don't override the dict.
    score_threshold: float = 1.5

    # Merge-pass threshold kept at 0.5 because segmentation changes
    # affect multiple characters at once and noise-level flips there
    # are more disruptive than within-segment flips.
    merge_threshold: float = 0.5

    # Cap per-segment candidate consideration — the Viterbi candidate
    # list can be long for short readings, and we don't need to score
    # deep entries.
    max_candidates_per_segment: int = 4

    # Add hiragana + katakana identity to the per-segment candidate
    # pool before LLM rescoring, so the LLM can pick kana for a
    # segment even when the dictionary only offered archaic kanji
    # (る → 縷 / 鏤 / 婁 / ...).
    #
    # Gated on the top dictionary entry's cost because a web-trained
    # LM has a systematic preference for hiragana over formal-writing
    # kanji (年/主に/等/及び) that we don't want to expose: adding ねん
    # to the 年 segment causes the LLM to flip to ねん in "1972年" even
    # though that's clearly a formal-writing context where kanji is
    # right. Only augment when the dict top pick looks archaic/rare.
    always_include_kana_forms: bool = True
    # Only augment when top dict entry cost > this. Common words (年,
    # 主, 等, 呼称) are cost 300-2000; boosted loanwords sit at 2500-
    # 2700; rare/archaic kanji like 縷, 鳴ん, 迸 are 3000+.
    # Setting the gate at 2500 means:
    #   * 縷, 鳴ん (cost 3000+) → augmented → LLM can fall back to kana
    #   * 年, 主に, 等 (cost 500-1500) → NOT augmented → kanji preserved
    #   * loanword entries (cost 2500-2700) → NOT augmented → kanji wins
    kana_augment_cost_threshold: int = 2500

    # Viterbi-confidence gate: skip LLM rescoring entirely for a
    # segment if the gap between the best and second-best candidate
    # is at least this many units of dictionary cost.
    #
    # 0 (default) = disabled -- the LLM evaluates every ambiguous
    # segment. Positive values restore the gate for speed:
    #   500  = LLM only sees Viterbi-ties
    #   2000 = LLM only sees near-exact ties
    # Available for users who prefer speed over thoroughness.
    viterbi_confidence_gap: int = 0

    # Prefix prepended to the whole surface before scoring, useful for
    # domain priming (e.g. "ニュース記事: "). Empty by default.
    prompt_prefix: str = ""

    # Also try merging adjacent segments to see if a single-unit
    # alternative scores higher. Captures segmentation errors the
    # Viterbi made at the lattice-level (e.g., the Viterbi split
    # "きのう" → き+のう where the merged 昨日 is obviously right).
    # Costs one LLM score call per mergeable pair, so for N segments
    # it's an extra O(N) calls on top of the within-segment pass.
    try_segment_merges: bool = True

    # When evaluating segment-merge alternatives, only consider
    # dictionary entries for the merged reading with cost at most this
    # much above the cheapest -- avoids paying LLM calls on a long
    # tail of rare proper-noun compounds that will never beat the
    # split.
    merge_candidate_cost_limit: int = 1500


@dataclass
class RescoreResult:
    """Output of a rescoring pass. ``overrides`` maps segment index to
    the chosen candidate index inside that segment's candidate list.
    Feed this into ComposerState.overrides to apply the rescoring.

    If ``alt_result`` is set, the rescorer decided a different
    segmentation (produced by merging / splitting segments) is
    preferable. Callers wanting a full segmentation switch should
    honor ``alt_result`` in place of the original ConversionResult;
    ``overrides`` then references the alt's candidates. When
    ``alt_result`` is None, ``overrides`` continues to refer to the
    original result."""

    overrides: dict[int, int] = field(default_factory=dict)
    surface: str = ""  # resulting surface after applying overrides
    llm_hits: int = 0  # how many segments the LLM changed
    alt_result: Optional[ConversionResult] = None  # segmentation override


class Rescorer:
    """Re-rank candidates inside each ConversionSegment using LLM scores.

    With ``dictionary`` supplied, also evaluates alternative
    segmentations -- specifically, merging adjacent segments when a
    single dictionary entry covers both readings. This catches
    Viterbi lattice errors that within-segment rescoring can't fix.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        config: Optional[RescoreConfig] = None,
        dictionary: Optional[Dictionary] = None,
    ) -> None:
        self.backend: Backend = backend or DummyBackend()
        self.config = config or RescoreConfig()
        self.dictionary = dictionary

    def rescore(
        self,
        frozen_prefix: str,
        result: ConversionResult,
        initial_overrides: Optional[dict[int, int]] = None,
    ) -> RescoreResult:
        """Run the LLM over ``result`` and return overrides that yield
        the LLM-preferred surface.

        When the backend isn't available, falls through with no
        changes — callers can treat this as a no-op without checking.
        """
        overrides: dict[int, int] = dict(initial_overrides or {})
        out = RescoreResult(overrides=overrides)
        if not self.backend.available or not result.segments:
            out.surface = result.surface_at(overrides)
            return out

        max_cands = self.config.max_candidates_per_segment
        threshold = self.config.score_threshold
        confidence_gap = self.config.viterbi_confidence_gap
        prefix_prompt = self.config.prompt_prefix
        hits = 0

        # Augment each segment's candidate list with hiragana and
        # katakana identity so the LLM can pick kana for a segment
        # even when the dictionary only offered archaic kanji
        # (る → 縷 / 鏤 / 婁 / ...). The augmented segments become a
        # new ConversionResult (``augmented_result``) that the composer
        # will receive via alt_result; the composer's Space-cycle
        # candidate list then includes these kana forms too.
        if self.config.always_include_kana_forms:
            augment_gate = self.config.kana_augment_cost_threshold
            augmented_segments: list[ConversionSegment] = []
            any_augmented = False
            for seg in result.segments:
                # Gate: only augment when the top dict entry looks
                # archaic/rare. Common-word segments (年, 主に, 等) have
                # cheap top entries and don't need kana augmentation;
                # augmenting them would let the web-trained LM's hiragana
                # bias flip formal-writing kanji (1972年→1972ねん,
                # 主に→おもに, 等→など) which is worse than the archaic
                # kanji we're trying to avoid.
                top_cost = (
                    seg.candidates[0].cost if seg.candidates else 99999
                )
                if top_cost <= augment_gate:
                    augmented_segments.append(seg)
                    continue
                aug_cands = list(seg.candidates)
                existing = {c.surface for c in aug_cands}
                added = False
                # Insert kana forms right after the top-``max_cands``
                # dict entries so the LLM's ``n_cands``-sized batch sees
                # them (appending at the end would hide them when dict
                # has more than max_cands entries).
                insert_pos = min(len(aug_cands), max_cands)
                kana_inserts: list[DictEntry] = []
                if seg.reading not in existing and seg.reading:
                    kana_inserts.append(DictEntry(
                        reading=seg.reading,
                        surface=seg.reading,
                        cost=4000,
                        source="llm-pool-hira",
                    ))
                from ..engine.dictionary import hiragana_to_katakana
                kata = hiragana_to_katakana(seg.reading)
                if kata != seg.reading and kata not in existing:
                    kana_inserts.append(DictEntry(
                        reading=seg.reading,
                        surface=kata,
                        cost=4100,
                        source="llm-pool-kata",
                    ))
                if kana_inserts:
                    aug_cands = (
                        aug_cands[:insert_pos]
                        + kana_inserts
                        + aug_cands[insert_pos:]
                    )
                    added = True
                if added:
                    any_augmented = True
                augmented_segments.append(ConversionSegment(
                    start=seg.start, end=seg.end,
                    reading=seg.reading, surface=seg.surface,
                    candidates=aug_cands,
                ))
            if any_augmented:
                working_result = ConversionResult(
                    reading=result.reading, segments=augmented_segments,
                )
            else:
                working_result = result
        else:
            working_result = result

        # Current decision for each segment (index into candidates).
        # Computed against the augmented segments so the indices match.
        choices: list[int] = [
            overrides.get(i, 0) for i in range(len(working_result.segments))
        ]

        for i, seg in enumerate(working_result.segments):
            # With augmentation, this segment may have had up to 2 kana
            # forms inserted at positions [max_cands:max_cands+2]. Cap
            # at max_cands + 2 so they're always inside the scoring
            # batch. Non-augmented segments cap at max_cands as usual.
            n_dict_kanji = sum(
                1 for c in seg.candidates[:max_cands + 2]
                if not c.source.startswith("llm-pool-")
            )
            n_augmented = len(seg.candidates[:max_cands + 2]) - n_dict_kanji
            pool_size = max_cands + n_augmented
            n_cands = min(len(seg.candidates), pool_size)
            if n_cands <= 1:
                continue
            # Viterbi-confidence gate: if the top dictionary cost beats
            # the runner-up by more than ``confidence_gap``, the
            # dictionary is already decisive. Skipping the LLM here
            # both speeds things up and prevents "style-preference"
            # flips like 見た↔観た or 違う↔ちがう that are all equally
            # valid but may deviate from the user's expected form.
            #
            # ``confidence_gap <= 0`` disables the gate entirely --
            # LLM is consulted on every multi-candidate segment.
            # Sorted order guarantees cand0.cost <= cand1.cost, so
            # ``gap >= 0`` is always true; treating 0 as "always skip"
            # would gate out the whole LLM pass.
            if confidence_gap > 0:
                cand0 = seg.candidates[0]
                cand1 = seg.candidates[1]
                if cand1.cost - cand0.cost >= confidence_gap:
                    continue
            # Build all K candidate surfaces at once and score them
            # in a single forward pass. This is the main CPU-speed
            # win: the tokenize + model forward is paid once per
            # segment instead of K times.
            prefix_full = prefix_prompt + frozen_prefix
            candidate_surfaces = [
                _build_surface(prefix_full, working_result.segments, choices, i, k)
                for k in range(n_cands)
            ]
            scores = self.backend.score_batch(candidate_surfaces)

            best_idx = choices[i]
            best_score = scores[best_idx] if best_idx < len(scores) else float("-inf")
            original_score = best_score
            for k in range(n_cands):
                if k == choices[i]:
                    continue
                if scores[k] > best_score:
                    best_score = scores[k]
                    best_idx = k
            if best_idx != choices[i] and (best_score - original_score) >= threshold:
                choices[i] = best_idx
                overrides[i] = best_idx
                hits += 1

        # Segmentation pass: generate alternative segmentations and
        # let the LLM pick the best one. Two strategies stacked:
        #   (a) merging adjacent segments (_try_merges)
        #   (b) shifting the boundary between adjacent segments ±1 or
        #       ±2 chars so the LLM also sees near-miss splits where
        #       Viterbi's dictionary-cost picked a wrong boundary
        #       (_try_boundary_shifts).
        #
        # Segmentation passes use their own ``merge_threshold``. It
        # can differ from the within-segment threshold, though in
        # practice we want segmentation changes to be at least as
        # conservative -- take ``max`` of the two so a user loosening
        # ``score_threshold`` doesn't accidentally open the floodgates
        # for multi-char segmentation flips.
        merge_threshold = max(threshold, self.config.merge_threshold)
        alt_result: Optional[ConversionResult] = None
        if self.config.try_segment_merges and self.dictionary is not None:
            working = working_result
            alt_result_merges, alt_hits_merges = self._try_merges(
                frozen_prefix=prefix_prompt + frozen_prefix,
                result=working,
                choices=choices,
                threshold=merge_threshold,
            )
            hits += alt_hits_merges
            if alt_result_merges is not None:
                working = alt_result_merges
                # Merges dropped overrides; start from [0]*N for the
                # shift pass.
                choices = [0] * len(working.segments)

            alt_result_shift, alt_hits_shift = self._try_boundary_shifts(
                frozen_prefix=prefix_prompt + frozen_prefix,
                result=working,
                choices=choices,
                threshold=merge_threshold,
            )
            hits += alt_hits_shift
            if alt_result_shift is not None:
                alt_result = alt_result_shift
            elif alt_result_merges is not None:
                alt_result = alt_result_merges

        # If we augmented candidates but didn't otherwise change
        # segmentation, still return the augmented result as alt_result
        # so the composer's Space-cycle list includes the kana forms
        # (the winning candidate might BE one of those kana entries
        # that wasn't in the original seg.candidates).
        if (
            alt_result is None
            and self.config.always_include_kana_forms
            and working_result is not result
        ):
            alt_result = working_result

        out.overrides = overrides
        out.surface = (alt_result or result).surface_at(overrides if alt_result is None else {})
        out.llm_hits = hits
        out.alt_result = alt_result
        return out

    def _try_merges(
        self,
        frozen_prefix: str,
        result: ConversionResult,
        choices: list[int],
        threshold: float,
    ) -> tuple[Optional[ConversionResult], int]:
        """Probe each adjacent pair (i, i+1) for a dictionary entry
        covering both readings. If LLM prefers the merged form over
        the split one by at least ``threshold`` nats, apply it.

        Returns (alt_result | None, hits). ``alt_result`` replaces the
        whole ConversionResult when segmentation changed; callers need
        to propagate it into the composer. Multiple merges compound:
        each accepted merge updates the working result and the loop
        continues from there. We do NOT iterate past a single
        left-to-right pass.
        """
        assert self.dictionary is not None
        segments = list(result.segments)
        if len(segments) < 2:
            return None, 0

        def _current_surface(segs: list[ConversionSegment], sel: list[int]) -> str:
            parts = []
            for i, s in enumerate(segs):
                k = sel[i] if i < len(sel) else 0
                if 0 <= k < len(s.candidates):
                    parts.append(s.candidates[k].surface)
                else:
                    parts.append(s.surface)
            return "".join(parts)

        cur_surface = _current_surface(segments, choices)
        base_score = self.backend.score(frozen_prefix + cur_surface)

        cost_limit = self.config.merge_candidate_cost_limit
        merges_applied = 0
        i = 0
        while i < len(segments) - 1:
            seg_a = segments[i]
            seg_b = segments[i + 1]
            merged_reading = seg_a.reading + seg_b.reading
            entries = sorted(
                self.dictionary.lookup(merged_reading),
                key=lambda e: e.cost,
            )
            if not entries:
                i += 1
                continue
            cheapest = entries[0].cost
            candidates_to_try = [
                e for e in entries if e.cost <= cheapest + cost_limit
            ][: self.config.max_candidates_per_segment]
            if not candidates_to_try:
                i += 1
                continue

            # Batch-score the whole set of merge candidates in one
            # forward pass. Same motivation as the within-segment
            # loop: one tokenize+forward vs N of them.
            pre = _current_surface(segments[:i], choices[:i])
            post = _current_surface(segments[i + 2:], choices[i + 2:])
            merge_surfaces = [
                frozen_prefix + pre + entry.surface + post
                for entry in candidates_to_try
            ]
            merge_scores = self.backend.score_batch(merge_surfaces)

            best_entry: Optional[DictEntry] = None
            best_score = base_score
            for entry, s in zip(candidates_to_try, merge_scores):
                if s - base_score >= threshold and s > best_score:
                    best_score = s
                    best_entry = entry

            if best_entry is None:
                i += 1
                continue

            # Accept the merge. Build a new ConversionSegment and
            # splice it into ``segments``.
            merged_seg = ConversionSegment(
                start=seg_a.start,
                end=seg_b.end,
                reading=merged_reading,
                surface=best_entry.surface,
                candidates=candidates_to_try,
            )
            segments = segments[:i] + [merged_seg] + segments[i + 2:]
            choices = choices[:i] + [0] + choices[i + 2:]
            cur_surface = _current_surface(segments, choices)
            base_score = best_score
            merges_applied += 1
            # Don't advance i -- another merge may now apply at the
            # same position (three-segment collapses).

        if merges_applied == 0:
            return None, 0
        new_result = ConversionResult(reading=result.reading, segments=segments)
        return new_result, merges_applied

    def _try_boundary_shifts(
        self,
        frozen_prefix: str,
        result: ConversionResult,
        choices: list[int],
        threshold: float,
    ) -> tuple[Optional[ConversionResult], int]:
        """For each adjacent pair of segments, try shifting the
        boundary between them by 1 or 2 reading-characters in either
        direction. If the shifted segmentation has valid dictionary
        entries for both halves AND the LLM scores the resulting
        surface higher than the current segmentation by ≥ threshold
        nats, adopt the shift.

        Complements _try_merges: merges consider the union of two
        segments as one unit; shifts consider two segments of
        different widths. Together they give LLM visibility into the
        main classes of Viterbi mis-splits.

        Accepts at most one shift per pair per pass -- compounding
        shifts across the sentence would blow up candidate count.
        """
        assert self.dictionary is not None
        if len(result.segments) < 2:
            return None, 0

        segments = list(result.segments)
        max_cands = self.config.max_candidates_per_segment

        # Helper to build the full current surface with the given
        # segments + choices.
        def _full_surface(segs: list[ConversionSegment], sel: list[int]) -> str:
            parts = []
            for idx, s in enumerate(segs):
                k = sel[idx] if idx < len(sel) else 0
                if 0 <= k < len(s.candidates):
                    parts.append(s.candidates[k].surface)
                else:
                    parts.append(s.surface)
            return "".join(parts)

        base_score = self.backend.score(frozen_prefix + _full_surface(segments, choices))
        shifts_applied = 0
        i = 0
        while i < len(segments) - 1:
            seg_a = segments[i]
            seg_b = segments[i + 1]
            combined = seg_a.reading + seg_b.reading
            n = len(combined)
            if n < 2:
                i += 1
                continue
            # Current boundary position within ``combined``.
            cur_split = len(seg_a.reading)
            # Candidate split positions: one or two chars either side
            # of the current split, clipped into [1, n-1].
            cand_splits = sorted({
                p for p in (cur_split - 2, cur_split - 1,
                             cur_split + 1, cur_split + 2)
                if 1 <= p <= n - 1
            })
            if not cand_splits:
                i += 1
                continue

            # Build batch of (alt_surface, new_seg_a_entry, new_seg_b_entry, split_pos)
            batch_items = []
            for split_pos in cand_splits:
                left_reading = combined[:split_pos]
                right_reading = combined[split_pos:]
                left_entries = sorted(
                    self.dictionary.lookup(left_reading),
                    key=lambda e: e.cost,
                )[:max_cands]
                right_entries = sorted(
                    self.dictionary.lookup(right_reading),
                    key=lambda e: e.cost,
                )[:max_cands]
                if not left_entries or not right_entries:
                    continue
                # Only try the top entry of each side for the shift
                # itself -- extra-candidate rescoring is done by the
                # within-segment pass on the winning shift.
                le = left_entries[0]
                re = right_entries[0]
                pre = _full_surface(segments[:i], choices[:i])
                post = _full_surface(segments[i + 2:], choices[i + 2:])
                alt_surface = pre + le.surface + re.surface + post
                batch_items.append((alt_surface, le, re, split_pos,
                                    left_entries, right_entries))

            if not batch_items:
                i += 1
                continue

            surfaces = [frozen_prefix + b[0] for b in batch_items]
            scores = self.backend.score_batch(surfaces)

            best_item = None
            best_score = base_score
            for item, s in zip(batch_items, scores):
                if s - base_score >= threshold and s > best_score:
                    best_score = s
                    best_item = item

            if best_item is None:
                i += 1
                continue

            _alt_surf, le, re, split_pos, left_ents, right_ents = best_item
            new_a = ConversionSegment(
                start=seg_a.start,
                end=seg_a.start + split_pos,
                reading=combined[:split_pos],
                surface=le.surface,
                candidates=left_ents,
            )
            new_b = ConversionSegment(
                start=seg_a.start + split_pos,
                end=seg_b.end,
                reading=combined[split_pos:],
                surface=re.surface,
                candidates=right_ents,
            )
            segments = segments[:i] + [new_a, new_b] + segments[i + 2:]
            choices = choices[:i] + [0, 0] + choices[i + 2:]
            base_score = best_score
            shifts_applied += 1
            i += 1  # don't re-test the same pair

        if shifts_applied == 0:
            return None, 0
        return (
            ConversionResult(reading=result.reading, segments=segments),
            shifts_applied,
        )


def _build_surface(
    prefix: str,
    segments: list[ConversionSegment],
    choices: list[int],
    target_idx: int,
    target_choice: int,
) -> str:
    """Return prefix + concatenated segment surfaces, with
    segment[target_idx] swapped to its target_choice candidate.
    Shared helper used by both the batched and the legacy
    single-score paths."""
    parts: list[str] = []
    for i, seg in enumerate(segments):
        if i == target_idx:
            k = target_choice
        else:
            k = choices[i]
        if 0 <= k < len(seg.candidates):
            parts.append(seg.candidates[k].surface)
        else:
            parts.append(seg.surface)
    return prefix + "".join(parts)


def _score_with_choice(
    backend: Backend,
    prefix: str,
    segments: list[ConversionSegment],
    choices: list[int],
    target_idx: int,
    target_choice: int,
) -> float:
    """Build the full surface with segment[target_idx] replaced by its
    target_choice'th candidate, prepend the frozen prefix, and return
    the backend's score."""
    parts: list[str] = []
    for i, seg in enumerate(segments):
        if i == target_idx:
            k = target_choice
        else:
            k = choices[i]
        if 0 <= k < len(seg.candidates):
            parts.append(seg.candidates[k].surface)
        else:
            parts.append(seg.surface)
    return backend.score(prefix + "".join(parts))
