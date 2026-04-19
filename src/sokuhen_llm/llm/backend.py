"""Backend protocols for the LLM rescorer.

A backend's one job is: given a string, return the natural-language
plausibility score (higher = more plausible). The rescorer uses the
score differences between candidate surfaces, so absolute scale
doesn't matter as long as it's consistent within one backend.

Ships with:
  * ``DummyBackend``  — zero-score, used as the fallback when no LLM
    is available or the user has disabled LLM rescoring. Lets the
    rest of the app be ignorant of whether LLM is installed.
  * ``HFBackend``     — HuggingFace transformers, loads a local
    causal LM and computes ``-loss * num_tokens`` as the score
    (i.e. total log-likelihood).
"""
from __future__ import annotations

import logging
from typing import Protocol


log = logging.getLogger(__name__)


class Backend(Protocol):
    """A log-likelihood provider.

    Implementations must be thread-safe for reads (the UI thread calls
    ``score`` while the composer may be mutating state). The simplest
    compliant implementation is stateless-after-init.
    """

    def score(self, text: str) -> float:
        """Return the LLM's log-likelihood of ``text``.

        Higher = more plausible under the model. Absolute scale is
        implementation-defined; the rescorer only uses differences.
        """
        ...

    @property
    def available(self) -> bool:
        """True if this backend is usable. A ``DummyBackend`` returns
        False so callers can bail out early; a real backend returns
        True once the model is loaded."""
        ...


class DummyBackend:
    """No-op backend — always returns 0.0.

    Used when the user has no LLM installed, the model weights are
    missing, or they explicitly disabled LLM rescoring. The rescorer
    detects this via ``available`` and takes a fast path that just
    returns the Viterbi result unchanged.
    """

    @property
    def available(self) -> bool:
        return False

    def score(self, text: str) -> float:  # noqa: ARG002
        return 0.0
