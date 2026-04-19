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

from ..engine.viterbi import ConversionResult, ConversionSegment
from .backend import Backend, DummyBackend


log = logging.getLogger(__name__)


@dataclass
class RescoreConfig:
    """Knobs for the rescoring pass."""

    # Log-prob delta required for LLM to override the Viterbi choice.
    # Lower = LLM more aggressive; higher = LLM more conservative.
    score_threshold: float = 0.5

    # Cap per-segment candidate consideration — the Viterbi candidate
    # list can be long for short readings, and we don't need to score
    # deep entries.
    max_candidates_per_segment: int = 4

    # Prefix prepended to the whole surface before scoring, useful for
    # domain priming (e.g. "ニュース記事: "). Empty by default.
    prompt_prefix: str = ""


@dataclass
class RescoreResult:
    """Output of a rescoring pass. ``overrides`` maps segment index to
    the chosen candidate index inside that segment's candidate list.
    Feed this into ComposerState.overrides to apply the rescoring."""

    overrides: dict[int, int] = field(default_factory=dict)
    surface: str = ""  # resulting surface after applying overrides
    llm_hits: int = 0  # how many segments the LLM changed


class Rescorer:
    """Re-rank candidates inside each ConversionSegment using LLM scores."""

    def __init__(
        self,
        backend: Optional[Backend] = None,
        config: Optional[RescoreConfig] = None,
    ) -> None:
        self.backend: Backend = backend or DummyBackend()
        self.config = config or RescoreConfig()

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
        prefix_prompt = self.config.prompt_prefix
        hits = 0

        for i, seg in enumerate(result.segments):
            n_cands = min(len(seg.candidates), max_cands)
            if n_cands <= 1:
                continue
            best_idx = choices[i]
            best_score = _score_with_choice(
                self.backend, prefix_prompt + frozen_prefix, result.segments,
                choices, i, best_idx,
            )
            original_score = best_score
            for k in range(n_cands):
                if k == choices[i]:
                    continue
                s = _score_with_choice(
                    self.backend, prefix_prompt + frozen_prefix, result.segments,
                    choices, i, k,
                )
                if s > best_score:
                    best_score = s
                    best_idx = k
            if best_idx != choices[i] and (best_score - original_score) >= threshold:
                choices[i] = best_idx
                overrides[i] = best_idx
                hits += 1

        out.overrides = overrides
        out.surface = result.surface_at(overrides)
        out.llm_hits = hits
        return out


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
