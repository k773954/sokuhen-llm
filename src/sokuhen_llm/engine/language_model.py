"""Lightweight bigram language model over dictionary surfaces.

We don't have a full mozc-style cost model, so we approximate:
- unigram cost derives from the dictionary's per-entry cost
- bigram cost is a small learned bonus/penalty between surface pairs that
  co-occur frequently in committed user text (see ``learning.py``)

The model is deliberately tiny and swappable. It's reasonable enough to
prefer common collocations ("今日 は", "する こと") once the user has
committed a few sentences, while starting from a sensible default on day one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# "Transition" cost when no bigram data exists. A bit higher than a confident
# match so that learned pairs can pull ahead; low enough that a fresh LM
# doesn't bias Viterbi significantly.
DEFAULT_BIGRAM_COST = 500

# A strong co-occurrence bonus (subtracted from cost). Capped so a single pair
# never completely dominates unigram costs.
MAX_BIGRAM_BONUS = 600


def grammatical_delta(prev: str, curr: str) -> int:
    """No fixed phrase priors.

    The runtime should rank candidates from dictionary costs, learned user
    bigrams, and the LLM rescorer. Hard-coded sentence/context rules are
    intentionally disabled so accuracy improvements do not come from
    memorizing individual examples.
    """
    return 0


@dataclass
class LanguageModel:
    """Surface-to-surface bigram scores.

    ``bigram_counts[(prev, curr)]`` holds how many times the sequence has been
    confirmed by the user. ``unigram_counts[surface]`` holds raw frequency
    used to normalize.
    """

    bigram_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    unigram_counts: dict[str, int] = field(default_factory=dict)

    # BOS token used as the previous surface at the start of a sentence.
    BOS: str = "<s>"

    def observe(self, surfaces: list[str]) -> None:
        """Record a committed sequence. Updates both unigram and bigram counts."""
        prev = self.BOS
        for surf in surfaces:
            self.unigram_counts[surf] = self.unigram_counts.get(surf, 0) + 1
            key = (prev, surf)
            self.bigram_counts[key] = self.bigram_counts.get(key, 0) + 1
            prev = surf

    def transition_cost(self, prev: str, curr: str) -> int:
        """Cost of transitioning from ``prev`` to ``curr`` surface.

        Lower is better. Uses only learned bigram evidence from the user's
        past commits. Pre-authored sentence/phrase priors are not applied;
        contextual disambiguation belongs to the LLM rescorer.
        """
        count = self.bigram_counts.get((prev, curr), 0)
        base = DEFAULT_BIGRAM_COST
        if count > 0:
            # prev_total must always be >= count (prev has appeared at least
            # as many times as the bigram). A bad save file could desync the
            # two tables; clamp defensively so a missing unigram doesn't
            # inflate the bonus to p≈1.
            prev_total = max(self.unigram_counts.get(prev, 0), count)
            prob = (count + 1) / (prev_total + 10)
            bonus = int(
                MAX_BIGRAM_BONUS * min(1.0, -math.log(max(1e-6, 1 - prob)) / 4.0)
            )
            base -= bonus
        base += grammatical_delta(prev, curr)
        return base

    def to_json(self) -> dict:
        return {
            "version": 1,
            "bigrams": [[p, c, n] for (p, c), n in self.bigram_counts.items()],
            "unigrams": list(self.unigram_counts.items()),
        }

    @classmethod
    def from_json(cls, data: dict) -> "LanguageModel":
        m = cls()
        for p, c, n in data.get("bigrams", []):
            m.bigram_counts[(p, c)] = int(n)
        for s, n in data.get("unigrams", []):
            m.unigram_counts[s] = int(n)
        return m
