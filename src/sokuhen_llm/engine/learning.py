"""Per-user learning store: remembers picks and updates a small LM.

Data is JSON on disk so it's easy to inspect, diff, and move between machines.
We deliberately keep it small and bounded: last N picks per reading, and
bigram counts capped at a sane max so a typo binge doesn't permanently bias
the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .language_model import LanguageModel


MAX_PICKS_PER_READING = 16
MAX_BIGRAM_COUNT = 50_000


@dataclass
class LearningStore:
    """Records user picks and wraps a mutable LanguageModel.

    ``picks[reading]`` is an ordered list of surfaces; the front is most
    recent. ``preferred_surface(reading)`` returns the front of the list.
    Bigram/unigram counts live on ``self.lm`` so the Viterbi converter
    picks them up directly.
    """

    picks: dict[str, list[str]] = field(default_factory=dict)
    lm: LanguageModel = field(default_factory=LanguageModel)
    path: Optional[Path] = None

    # --- recording -------------------------------------------------------

    def record_pick(self, reading: str, surface: str) -> None:
        if not reading or not surface:
            return
        lst = self.picks.setdefault(reading, [])
        if surface in lst:
            lst.remove(surface)
        lst.insert(0, surface)
        if len(lst) > MAX_PICKS_PER_READING:
            del lst[MAX_PICKS_PER_READING:]

    def preferred_surface(self, reading: str) -> Optional[str]:
        lst = self.picks.get(reading)
        return lst[0] if lst else None

    # --- persistence -----------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "LearningStore":
        store = cls(path=path)
        if not path.exists():
            return store
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return store
        store.picks = {k: list(v) for k, v in data.get("picks", {}).items()}
        store.lm = LanguageModel.from_json(data.get("lm", {}))
        return store

    def save(self, path: Optional[Path] = None) -> None:
        target = path or self.path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {"picks": self.picks, "lm": self.lm.to_json()}
        # Cap bigram counts to prevent unbounded growth.
        lm_json = data["lm"]
        for triple in lm_json.get("bigrams", []):
            if triple[2] > MAX_BIGRAM_COUNT:
                triple[2] = MAX_BIGRAM_COUNT
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
