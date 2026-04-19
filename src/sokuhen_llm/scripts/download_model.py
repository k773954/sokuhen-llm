"""One-shot: download the LLM weights into the local ``models/`` cache.

After this runs, sokuhen-llm can load the model with
``local_files_only=True`` on subsequent launches — no further network
access, no telemetry.

Usage:

    python -m sokuhen_llm.scripts.download_model
    python -m sokuhen_llm.scripts.download_model --model cyberagent/open-calm-small

Default model: ``rinna/japanese-gpt2-small`` (110M params, MIT licence,
~440 MB). See README for alternative Japanese LMs that work with this
rescorer.
"""
from __future__ import annotations

import argparse
import sys

from ..paths import models_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download LLM weights for sokuhen-llm")
    parser.add_argument(
        "--model",
        default="rinna/japanese-gpt2-small",
        help="HuggingFace model id (default: rinna/japanese-gpt2-small)",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination cache dir (default: repo's models/ directory)",
    )
    args = parser.parse_args(argv)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print(
            "ERROR: transformers not installed. Run:\n"
            "    python -m pip install transformers torch",
            file=sys.stderr,
        )
        return 1

    dest = args.dest or str(models_dir())
    print(f"Downloading {args.model} into {dest} ...", file=sys.stderr)

    # Trigger download via from_pretrained; weights land in the HF cache
    # inside ``dest``. The backend will find them again via the same
    # cache_dir on load.
    AutoTokenizer.from_pretrained(args.model, cache_dir=dest)
    AutoModelForCausalLM.from_pretrained(args.model, cache_dir=dest)

    print(f"Done. sokuhen-llm will load from {dest} with no network access.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
