"""Local-only LLM rescoring layer for sokuhen-llm.

Exposes two things:

* ``Rescorer`` — takes a Viterbi ``ConversionResult`` plus a frozen
  prefix string, picks the best candidate inside each segment using
  LLM log-likelihood of the resulting surface sequence. Zero external
  API calls — the model is loaded from local weights.

* ``HFBackend`` / ``DummyBackend`` — pluggable scoring backends.
  ``HFBackend`` loads a HuggingFace causal LM (default:
  ``rinna/japanese-gpt2-small``, ~440 MB, 110M params, MIT licence)
  and runs inference on CPU via transformers + PyTorch.
  ``DummyBackend`` is a no-op used when LLM rescoring is disabled or
  when transformers isn't installed.
"""
from __future__ import annotations

from .backend import Backend, DummyBackend
from .rescorer import Rescorer

__all__ = ["Backend", "DummyBackend", "Rescorer"]
