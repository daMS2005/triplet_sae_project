#!/usr/bin/env python3
"""Compare predicted triples against reference triples.

This script is the stable public entrypoint for evaluation. It delegates to the
project's richer comparison implementation in `scripts/10_compare_triplet_extractors.py`
while presenting a cleaner top-level command for common use.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    target = Path(__file__).with_name("10_compare_triplet_extractors.py")
    try:
        namespace = runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return int(namespace.get("__return_code__", 0) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
