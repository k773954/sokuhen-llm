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
    # Lower = LLM more aggressive; higher = LLM more conservative.
    # At 1.5 nats the LLM needs to be ~4.5× more confident in the
    # alternative before overriding — high enough that style-preference
    # flips (観た vs 見た, ちがう vs 違う) stop drowning out the real
    # wins (事故 vs 自己, 使用 vs しよう).
    score_threshold: float = 1.5

    # Cap per-segment candidate consideration — the Viterbi candidate
    # list can be long for short readings, and we don't need to score
    # deep entries.
    max_candidates_per_segment: int = 4

    # Viterbi-confidence gate: skip LLM rescoring entirely for a
    # segment if the gap between the best and second-best candidate
    # is at least this many units of dictionary cost.
    #
    # 0 (default) = disabled -- the LLM evaluates every ambiguous
    # segment. Matches the user's "LLMモードの時はベースラインはなし"
    # request: when LLM is enabled, its judgement is authoritative,
    # not a sometimes-consulted second opinion.
    #
    # Positive values restore the gate for speed: 500 was the
    # previous default (LLM only sees Viterbi-ties), 2000 = LLM
    # only sees near-exact ties. Available for users who prefer
    # speed over thoroughness.
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

        # Current decision for each segment (index into candidates).
        choices: list[int] = [
            overrides.get(i, 0) for i in range(len(result.segments))
        ]
        max_cands = self.config.max_candidates_per_segment
        threshold = self.config.score_threshold
        confidence_gap = self.config.viterbi_confidence_gap
        prefix_prompt = self.config.prompt_prefix
        hits = 0

        for i, seg in enumerate(result.segments):
            n_cands = min(len(seg.candidates), max_cands)
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
                _build_surface(prefix_full, result.segments, choices, i, k)
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

        # Segmentation pass: try merging adjacent segments.
        alt_result: Optional[ConversionResult] = None
        if self.config.try_segment_merges and self.dictionary is not None:
            alt_result, alt_hits = self._try_merges(
                frozen_prefix=prefix_prompt + frozen_prefix,
                result=result,
                choices=choices,
                threshold=threshold,
            )
            hits += alt_hits

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
