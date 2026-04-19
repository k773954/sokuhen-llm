"""Viterbi converter: hiragana reading -> best kana/kanji sequence.

Builds a lattice where each node is a dictionary entry covering a span of the
input, plus fallback single/multi-character kana nodes when the dictionary
can't cover a position. Finds the lowest-cost path through the lattice with
a standard forward DP, using the bigram language model for transition costs.

For multiple candidates (Space cycling in the UI), we also return per-segment
alternatives sorted by unigram cost within the winning segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .dictionary import DictEntry, Dictionary, hiragana_to_katakana
from .language_model import LanguageModel


@dataclass(frozen=True)
class ConversionSegment:
    """One segment (文節-like unit) of the conversion output.

    ``reading`` is the hiragana input covered. ``surface`` is the currently
    chosen surface form. ``candidates`` are alternative surfaces sorted best
    first, used for Space cycling.
    """

    start: int
    end: int  # exclusive, into the original reading
    reading: str
    surface: str
    candidates: list[DictEntry]

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ConversionResult:
    """Output of ``Converter.convert``."""

    reading: str
    segments: list[ConversionSegment]

    @property
    def surface_text(self) -> str:
        return "".join(s.surface for s in self.segments)

    def surface_at(self, override: dict[int, int]) -> str:
        """Surface text with selected candidate index per segment.

        ``override`` maps segment index -> candidate index. Missing entries
        fall back to the segment's default (candidate 0).
        """
        parts = []
        for i, seg in enumerate(self.segments):
            idx = override.get(i, 0)
            if 0 <= idx < len(seg.candidates):
                parts.append(seg.candidates[idx].surface)
            else:
                parts.append(seg.surface)
        return "".join(parts)


# --- internal lattice ---------------------------------------------------


@dataclass
class _Node:
    end: int
    entry: DictEntry
    best_cost: float = float("inf")
    back_end: int = -1
    back_node: int = -1  # index into the node list ending at that position


# Upper bound on single-unit lengths for fallback when no dict match exists.
_FALLBACK_MAX = 3


# --- katakana auto-detection -----------------------------------------
#
# Characters that strongly signal a katakana loanword. When a reading run
# contains one of these, it's very likely the user wants katakana output
# rather than kanji. The Viterbi layer uses this to emit an auto-generated
# katakana entry covering the whole foreign-looking run.
_STRONG_FOREIGN = frozenset("ー")

# "Foreign-looking" digraphs (lead char + following char) rare in native
# Japanese. If the reading contains one of these pairs at consecutive
# positions, the surrounding run is likely a loanword. Listed as startwith
# checks on a 2-char window.
_FOREIGN_DIGRAPHS = frozenset([
    # u-column + small vowel (ウィ, ウェ, ウォ)
    "うぃ", "うぇ", "うぉ",
    # f-row (ファ, フィ, フェ, フォ)
    "ふぁ", "ふぃ", "ふぇ", "ふぉ",
    # v-row
    "ゔぁ", "ゔぃ", "ゔぇ", "ゔぉ",
    # t/d-column small-i/u (ティ, トゥ, ディ, ドゥ)
    "てぃ", "とぅ", "でぃ", "どぅ",
    # ch/sh/j with small-e (チェ, シェ, ジェ)
    "ちぇ", "しぇ", "じぇ",
    # ts-row small vowels (ツァ, ツェ, ツォ)
    "つぁ", "つぇ", "つぉ",
    # k/g with small-e/o (ケェ, コォ — rare but foreign)
    "くぉ",
])

# Any hiragana char that could plausibly be part of a katakana loanword.
# Used to define the bounds of the "foreign run" once we've decided a run
# is foreign.
_KATAKANA_CAPABLE = (
    set("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわ")
    | set("がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ")
    | set("ぁぃぅぇぉゃゅょっゔ")
    | set("ー")
)

# Common single-character Japanese particles. Hitting any of these ends
# the katakana run — a particle after a foreign word marks the word's
# boundary ("コーヒー" + "を" + "飲む").
_PARTICLE_BREAKS = frozenset("をのにはがともやへかよねわでんぞぜばぎくけ")


_MAX_FOREIGN_RUN = 6  # cap the scan distance — longer runs likely span words


def _foreign_run(reading: str, start: int) -> int:
    """If ``reading[start:]`` starts a "foreign-looking" run, return the
    exclusive end index. Otherwise return ``start`` (no run).

    Algorithm:
      1. Scan ahead over katakana-capable characters looking for a
         foreign signal (ー, or a foreign digraph like ふぁ/ゔぃ/ちぇ/
         とぅ). Limit search to ``_MAX_FOREIGN_RUN`` chars.
      2. If no signal found before a particle break or non-capable
         char, return ``start`` (no run).
      3. Otherwise extend past the signal until a particle break to
         include the remainder of the foreign word, then return.

    The run intentionally stops at particles so the auto-generated
    katakana entry doesn't swallow following particles / verbs
    (コーヒーを飲む → コーヒー + を + 飲む, not コーヒーヲノム).
    """
    n = len(reading)
    if start >= n:
        return start
    if reading[start] not in _KATAKANA_CAPABLE:
        return start

    # First pass: scan forward looking for a foreign signal (ー or
    # foreign digraph) within the search cap. Stop at particle breaks —
    # "の" between two words (首位のチーム) means the run shouldn't
    # bridge the particle, and the cost of missing mid-word に in
    # モニター is acceptable (we rely on the kata-dict extraction
    # from SKK-JISYO.L for that; モニター is explicitly registered).
    signal_end = -1
    limit = min(n, start + _MAX_FOREIGN_RUN)
    k = start
    while k < limit:
        c = reading[k]
        if c not in _KATAKANA_CAPABLE:
            break
        if k > start and c in _PARTICLE_BREAKS:
            break
        if c in _STRONG_FOREIGN:
            signal_end = k + 1
            break
        if k + 1 < limit and reading[k : k + 2] in _FOREIGN_DIGRAPHS:
            signal_end = k + 2
            break
        k += 1

    if signal_end < 0:
        return start  # no foreign signal

    # Second pass: extend from signal_end until a particle break or a
    # non-katakana-capable char. The signal's presence confirms we're
    # inside a foreign word, so a particle char after the signal is the
    # natural word boundary (コーヒー + を, モニター + が).
    end = signal_end
    while end < n:
        c = reading[end]
        if c not in _KATAKANA_CAPABLE:
            break
        if c in _PARTICLE_BREAKS:
            break
        end += 1

    if end - start < 2:
        return start

    return end


class Converter:
    """Hiragana -> best surface conversion via Viterbi over a word lattice."""

    def __init__(
        self,
        dictionary: Dictionary,
        language_model: Optional[LanguageModel] = None,
        *,
        max_candidates_per_segment: int = 10,
    ) -> None:
        self.dict = dictionary
        self.lm = language_model or LanguageModel()
        self.max_candidates = max_candidates_per_segment

    # ----------------------------------------------------------------

    def convert(self, reading: str, *, bos: Optional[str] = None) -> ConversionResult:
        """Convert a hiragana reading into segments with best surface choices.

        ``bos`` overrides the sentinel surface used as the left context for
        the first segment. Pass the preceding surface (e.g. the last word of
        already-frozen text) so the LM's bigram lookup sees a realistic
        left context; omit for true start-of-input. Previously this was
        done by swapping ``self.lm.BOS`` around the call, which was a
        hidden-side-effect footgun.

        Algorithm (compact Viterbi over words):
          - For each start position i, enumerate dictionary word lengths at i.
            If none exist, add a fallback kana node of length 1..FALLBACK_MAX.
          - For each node (i, j, entry), the cost reaching j via this node is
            min over all nodes ending at i of: their best_cost + transition(LM)
            + entry.cost.
          - Backtrack from position len(reading) to recover segments.
        """
        n = len(reading)
        if n == 0:
            return ConversionResult(reading="", segments=[])

        sentinel = bos if bos is not None else self.lm.BOS
        # nodes_ending[pos] is the list of nodes whose end == pos. Each node is
        # a (entry, best_cost, back_end, back_nodeidx) tuple; we use _Node.
        nodes_ending: list[list[_Node]] = [[] for _ in range(n + 1)]
        # Sentinel BOS at position 0.
        bos_node = _Node(end=0, entry=DictEntry(reading="", surface=sentinel, cost=0), best_cost=0.0)
        nodes_ending[0].append(bos_node)

        for i in range(n):
            prev_nodes = nodes_ending[i]
            if not prev_nodes:
                continue
            # Collect entries that can start at i.
            candidates: list[tuple[int, DictEntry]] = []
            lens = self.dict.candidate_lengths_at(reading, i)
            for ln in lens:
                for e in self.dict.lookup(reading[i : i + ln]):
                    candidates.append((ln, e))

            # Always ensure a fallback path covering at least one character
            # so the algorithm can complete even for unknown readings.
            # Only length-1 fallback: a longer fallback like "、おと" would
            # otherwise swallow the next-position dictionary word as part of
            # a single cheap-per-char fallback segment. Length-1 forces
            # Viterbi to pay a transition cost at every step, which lets
            # genuine dictionary compounds starting one position over win.
            if not lens:
                sub = reading[i : i + 1]
                for e in Dictionary.generate_fallback(sub):
                    candidates.append((1, e))
            else:
                # Also add a length-1 fallback so the lattice stays dense and
                # unusual segmentations remain reachable.
                sub = reading[i : i + 1]
                if not self.dict.contains(sub):
                    for e in Dictionary.generate_fallback(sub):
                        candidates.append((1, e))

            # Digit passthrough: if position i starts a run of ASCII digits,
            # add a single cheap passthrough entry covering the whole run.
            # SKK has entries like "15"→"十五" that would otherwise win over
            # per-digit length-1 fallbacks, since one segment with 3000 cost
            # beats two length-1 fallbacks at 8000 each. A low-cost single
            # passthrough entry prevents kanji-digit substitution.
            ch = reading[i]
            if ch.isascii() and ch.isdigit():
                j = i
                while j < n and reading[j].isascii() and reading[j].isdigit():
                    j += 1
                run = reading[i:j]
                candidates.append((
                    j - i,
                    DictEntry(reading=run, surface=run, cost=500, source="digit"),
                ))

            # Katakana auto-detection: last-resort fallback for
            # loanwords that aren't in FOREIGN_WORDS or the kata-dict
            # extraction. Fires only on short runs (≤6 chars) that
            # contain a foreign signal (ー or foreign digraph). Cost is
            # set above typical dict compounds (3200) so kanji wins by
            # default; katakana comes through when no compound covers
            # the reading (e.g., a rare loanword).
            run_end = _foreign_run(reading, i)
            if run_end > i + 1:
                run = reading[i:run_end]
                kata = hiragana_to_katakana(run)
                if kata != run:
                    candidates.append((
                        run_end - i,
                        DictEntry(reading=run, surface=kata, cost=3200, source="kata-auto"),
                    ))

            for ln, entry in candidates:
                j = i + ln
                best: Optional[_Node] = None
                best_cost = float("inf")
                best_back_idx = -1
                for bi, prev in enumerate(prev_nodes):
                    trans = self.lm.transition_cost(prev.entry.surface, entry.surface)
                    cost = prev.best_cost + trans + entry.cost
                    if cost < best_cost:
                        best_cost = cost
                        best = prev
                        best_back_idx = bi
                if best is None:
                    continue
                node = _Node(
                    end=j,
                    entry=entry,
                    best_cost=best_cost,
                    back_end=i,
                    back_node=best_back_idx,
                )
                nodes_ending[j].append(node)

        # Backtrack from position n: find lowest-cost node ending exactly there.
        end_nodes = nodes_ending[n]
        if not end_nodes:
            # Should only happen if the reading contains characters we can't
            # even fallback-represent; degrade to raw passthrough.
            return ConversionResult(
                reading=reading,
                segments=[
                    ConversionSegment(
                        start=0,
                        end=n,
                        reading=reading,
                        surface=reading,
                        candidates=[DictEntry(reading=reading, surface=reading, cost=9999, source="passthrough")],
                    )
                ],
            )

        best_idx = min(range(len(end_nodes)), key=lambda i: end_nodes[i].best_cost)
        path: list[_Node] = []
        cur_end = n
        cur_idx = best_idx
        while cur_end > 0:
            node = nodes_ending[cur_end][cur_idx]
            path.append(node)
            cur_end, cur_idx = node.back_end, node.back_node
        path.reverse()

        segments: list[ConversionSegment] = []
        for node in path:
            start = node.back_end
            end = node.end
            reading_slice = reading[start:end]
            # Collect alternative candidates for this span for UI cycling.
            alts = self.alternatives(reading_slice, node.entry)
            segments.append(
                ConversionSegment(
                    start=start,
                    end=end,
                    reading=reading_slice,
                    surface=node.entry.surface,
                    candidates=alts,
                )
            )
        return ConversionResult(reading=reading, segments=segments)

    # ----------------------------------------------------------------

    def alternatives(self, reading_slice: str, chosen: DictEntry) -> list[DictEntry]:
        """Alternatives for a single span. Order:
            1. The Viterbi-chosen surface (chosen).
            2. Hiragana and katakana versions of the reading (promoted early
               because short-reading kanji lists tend to be long, and users
               often want a plain kana fallback near the top — especially
               for single-character readings).
            3. Remaining dictionary entries, sorted by cost.
        """
        entries = sorted(
            (e for e in self.dict.lookup(reading_slice) if e.surface != chosen.surface),
            key=lambda e: e.cost,
        )

        hira = DictEntry(
            reading=reading_slice, surface=reading_slice, cost=8000, source="fallback"
        )
        kata_surface = hiragana_to_katakana(reading_slice)
        kata = (
            DictEntry(
                reading=reading_slice, surface=kata_surface, cost=9000, source="fallback"
            )
            if kata_surface != reading_slice
            else None
        )

        seen: set[str] = {chosen.surface}
        result: list[DictEntry] = [chosen]

        def _add(entry: DictEntry | None) -> None:
            if entry is None or entry.surface in seen:
                return
            result.append(entry)
            seen.add(entry.surface)

        # For short readings, kana fallbacks jump to the front (positions 2-3)
        # so the user can reach them with one or two Space presses.
        if len(reading_slice) <= 2:
            _add(hira)
            _add(kata)
            for e in entries:
                _add(e)
        else:
            # For longer readings, kanji dict hits are usually the right
            # answer — kana fallbacks slide to after the first couple of
            # alternatives.
            top = entries[:3]
            rest = entries[3:]
            for e in top:
                _add(e)
            _add(hira)
            _add(kata)
            for e in rest:
                _add(e)

        return result[: self.max_candidates]
