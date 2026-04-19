"""Download open-source dictionary data.

By default fetches SKK-JISYO.L (GPLv2+) from openskk/dict mirror. Output goes to
the repo-level ``data/`` directory. The file is gzipped EUC-JP originally; we
decode on load rather than during download.
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

from ..paths import data_dir


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    gzipped: bool
    encoding: str
    license: str


_RAW_BASE = "https://raw.githubusercontent.com/skk-dev/dict/master"


def _skk(name: str, license_: str = "GPLv2+") -> Source:
    return Source(
        name=name,
        url=f"{_RAW_BASE}/{name}",
        filename=name,
        gzipped=False,
        encoding="euc_jp",
        license=license_,
    )


SOURCES: dict[str, Source] = {
    "skk-l": _skk("SKK-JISYO.L"),
    "skk-edict": _skk("SKK-JISYO.edict"),
    "skk-edict2": _skk("SKK-JISYO.edict2"),
    "skk-jinmei": _skk("SKK-JISYO.jinmei"),
    "skk-geo": _skk("SKK-JISYO.geo"),
    "skk-fullname": _skk("SKK-JISYO.fullname"),
}


def fetch(source: Source, dest: Path, *, timeout: int = 60) -> Path:
    out_path = dest / source.filename
    if out_path.exists():
        print(f"[skip] {source.filename} already exists at {out_path}", file=sys.stderr)
        return out_path

    print(f"[get ] {source.url}", file=sys.stderr)
    resp = requests.get(source.url, timeout=timeout)
    resp.raise_for_status()
    raw = resp.content
    if source.gzipped:
        raw = gzip.decompress(raw)

    out_path.write_bytes(raw)
    print(
        f"[ok  ] {source.filename} ({len(raw):,} bytes, encoding={source.encoding}, license={source.license})",
        file=sys.stderr,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download dictionaries for sokuhen")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(SOURCES.keys()),
        help="Dictionaries to fetch (default: skk-l only — smallest useful set)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: repo data/)",
    )
    args = parser.parse_args(argv)

    dest = args.dest or data_dir()
    dest.mkdir(parents=True, exist_ok=True)

    keys = args.only or ["skk-l", "skk-edict2", "skk-jinmei"]
    for key in keys:
        fetch(SOURCES[key], dest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
