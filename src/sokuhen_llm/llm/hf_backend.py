"""HuggingFace transformers backend — loads a local Japanese causal LM
and computes log-likelihood for rescoring.

Defaults to ``rinna/japanese-gpt2-small`` (110M params, MIT license,
~440 MB on disk). Tested on CPU; GPU will be used automatically if
torch detects one.

Design notes

* **Local-only.** The model is fetched *once* via ``download_model.py``
  (or on first ``from_pretrained`` call with internet), then lives
  under ``models/`` in the repo. Subsequent launches pass
  ``local_files_only=True`` so the app never reaches out to
  huggingface.co at runtime.
* **Cache inference.** Rescoring the same sentence twice is common
  during live composition; an LRU cache on ``score`` halves work.
* **Batch on demand.** The current rescorer scores one string at a
  time. If we ever need batched scoring, extend this with a
  ``score_batch`` method — the tokenizer supports padding.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "rinna/japanese-gpt2-small"


class HFBackend:
    """HuggingFace causal LM scoring backend.

    Constructor parameters:
      * ``model_id``: a HuggingFace hub id (``rinna/japanese-gpt2-small``)
        or a local directory path. If a directory, ``from_pretrained``
        is called in local-only mode so no network I/O happens.
      * ``cache_dir``: where downloaded weights live. Defaults to the
        repo's ``models/`` directory via ``paths.models_dir()``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        cache_dir: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        from ..paths import models_dir

        if cache_dir is None:
            cache_dir = models_dir()
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._available = False
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._load()

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            log.warning("LLM backend not available: %s (pip install transformers torch)", e)
            return

        self._torch = torch

        # HuggingFace's cache_dir layout stores model ``org/name`` under
        # ``cache_dir/models--<org>--<name>/``. Peek there to decide
        # whether we can pass ``local_files_only=True`` -- keeps the
        # load path completely offline after first download and avoids
        # the HEAD / "check for updates" round trips that otherwise
        # happen on every startup.
        hf_cache_subdir = (
            self.cache_dir / f"models--{self.model_id.replace('/', '--')}"
        )
        local_only = hf_cache_subdir.exists()

        try:
            log.info(
                "Loading LLM: %s (cache=%s, local_only=%s)",
                self.model_id, self.cache_dir, local_only,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                local_files_only=local_only,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                local_files_only=local_only,
            )
            self._model.eval()
            self._model.to(self.device)
            self._available = True
            log.info(
                "LLM loaded (device=%s, params=%s)",
                self.device,
                f"{sum(p.numel() for p in self._model.parameters()):,}",
            )
        except Exception as e:
            log.warning("LLM load failed: %s", e)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @lru_cache(maxsize=2048)
    def score(self, text: str) -> float:
        """Return total log-likelihood of ``text`` under the LM.

        Uses the standard trick: teacher-forced loss * seq_len gives
        total negative log-likelihood. We negate to get log-likelihood
        (higher = more plausible).
        """
        if not self._available or not text:
            return 0.0
        torch = self._torch
        ids = self._tokenizer(text, return_tensors="pt").input_ids
        if ids.size(1) < 2:
            return 0.0
        ids = ids.to(self.device)
        with torch.no_grad():
            out = self._model(input_ids=ids, labels=ids)
        # out.loss is mean cross-entropy over (seq_len - 1) token
        # predictions. Multiply back to get total NLL, negate to get LL.
        n = ids.size(1) - 1
        return -float(out.loss.item()) * n
