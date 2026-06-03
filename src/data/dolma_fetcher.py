"""Utilities for collecting text documents from the Dolma dataset.

Dolma data is hosted at olmo-data.org as gzipped JSONL shards.
The Hugging Face repo (allenai/dolma) only contains URL lists pointing to those shards.
This module downloads URL lists from HF Hub, then streams documents directly from
the shard URLs — no HF datasets loading script required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise ImportError(
        "The 'huggingface_hub' package is required: pip install huggingface_hub"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOLMA_REPO = "allenai/dolma"
DOLMA_URL_FILE = "urls/v1_5-sample.txt"  # use sample list; swap for v1_5.txt for full corpus

# Source names are the subdirectory component of each shard URL.
DOLMA_SOURCES = (
    "books",
    "c4",
    "cc_en_head",
    "cc_en_middle",
    "cc_en_tail",
    "pes2o",
    "reddit",
    "stack",
    "wiki",
)

_USER_AGENT = "triplet_sae_project/0.1 (contact: you@example.com)"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Request config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DolmaFetchRequest:
    """Configuration for a single Dolma fetch operation.

    Args:
        max_documents:   Maximum number of documents to return.
        sources:         Restrict to these source subdirectories (e.g. ``["wiki", "books"]``).
                         Pass ``None`` to sample across all available sources.
        shuffle_urls:    Shuffle the list of shard URLs before streaming.
        random_seed:     Seed used when *shuffle_urls* is True.
        min_text_length: Skip documents whose cleaned text is shorter than this.
        url_file:        HF Hub path to the URL list file (default: v1_5 sample).
        hf_token:        Hugging Face access token (falls back to HF_TOKEN env var).
        timeout_seconds: HTTP timeout per request.
    """

    max_documents: int = 100
    sources: list[str] | None = None
    shuffle_urls: bool = True
    random_seed: int | None = None
    min_text_length: int = 50
    url_file: str = DOLMA_URL_FILE
    hf_token: str | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_documents <= 0:
            raise ValueError("max_documents must be > 0")
        if self.min_text_length < 0:
            raise ValueError("min_text_length must be >= 0")
        if self.sources is not None:
            unknown = [s for s in self.sources if s not in DOLMA_SOURCES]
            if unknown:
                raise ValueError(
                    f"Unknown Dolma source(s): {unknown}. "
                    f"Known sources: {list(DOLMA_SOURCES)}"
                )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class DolmaClient:
    """Streams Dolma documents directly from olmo-data.org shard URLs."""

    def __init__(
        self,
        hf_token: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._token = (
            hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        self.timeout = timeout_seconds

    # ------------------------------------------------------------------
    # URL list
    # ------------------------------------------------------------------

    def get_shard_urls(
        self,
        url_file: str,
        sources: list[str] | None,
    ) -> list[str]:
        """Download the URL list from HF Hub and filter by source."""
        local_path = hf_hub_download(
            DOLMA_REPO,
            url_file,
            repo_type="dataset",
            token=self._token,
        )
        with open(local_path, encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        if sources:
            # Each URL looks like: https://olmo-data.org/.../SOURCE/filename.json.gz
            # The source is the second-to-last path component.
            urls = [u for u in urls if _url_source(u) in sources]

        return urls

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _stream_shard(self, url: str) -> Iterator[dict]:
        """Yield raw dicts from a single gzipped JSONL shard URL."""
        headers = {"User-Agent": _USER_AGENT}
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=self.timeout) as resp:
                with gzip.open(resp, "rt", encoding="utf-8") as gz:
                    for line in gz:
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue
        except (HTTPError, URLError) as exc:
            print(f"Warning: could not fetch shard {url}: {exc}", file=sys.stderr)

    def iter_documents(
        self,
        url_file: str,
        sources: list[str] | None,
        shuffle_urls: bool,
        random_seed: int | None,
        min_text_length: int,
    ) -> Iterator[dict]:
        """Yield cleaned document dicts.

        Each dict has keys:
            ``doc_id``, ``title``, ``source``, ``source_url``,
            ``text``, ``created``, ``retrieved_at``
        """
        urls = self.get_shard_urls(url_file, sources)
        if not urls:
            raise ValueError("No shard URLs matched the requested sources.")

        if shuffle_urls:
            rng = random.Random(random_seed)
            rng.shuffle(urls)

        retrieved_at = datetime.now(timezone.utc).isoformat()

        for url in urls:
            for row in self._stream_shard(url):
                text = _clean_text(row.get("text") or "")
                if len(text) < min_text_length:
                    continue
                metadata: dict = row.get("metadata") or {}
                yield {
                    "doc_id": str(row.get("id") or ""),
                    "title": str(metadata.get("title") or ""),
                    "source": str(row.get("source") or _url_source(url)),
                    "source_url": str(metadata.get("url") or ""),
                    "text": text,
                    "created": str(row.get("created") or row.get("added") or ""),
                    "retrieved_at": retrieved_at,
                }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _url_source(url: str) -> str:
    """Extract source subdirectory name from a shard URL."""
    # e.g. https://olmo-data.org/dolma-v1_5r1/wiki/en_simple_wiki_v0-0000.json.gz → "wiki"
    parts = url.rstrip("/").split("/")
    return parts[-2] if len(parts) >= 2 else ""


# ---------------------------------------------------------------------------
# Top-level fetch function
# ---------------------------------------------------------------------------

def fetch_documents(request: DolmaFetchRequest) -> list[dict]:
    """Fetch up to *request.max_documents* documents from Dolma.

    Example::

        docs = fetch_documents(DolmaFetchRequest(max_documents=50, sources=["wiki"]))
    """
    client = DolmaClient(
        hf_token=request.hf_token,
        timeout_seconds=request.timeout_seconds,
    )

    results: list[dict] = []
    for doc in client.iter_documents(
        url_file=request.url_file,
        sources=request.sources,
        shuffle_urls=request.shuffle_urls,
        random_seed=request.random_seed,
        min_text_length=request.min_text_length,
    ):
        results.append(doc)
        if len(results) >= request.max_documents:
            break

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch documents from the Dolma dataset and write them as JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/dolma.jsonl"),
        metavar="FILE",
        help="Output JSONL file (default: data/raw/dolma.jsonl).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=100,
        metavar="N",
        help="Maximum number of documents to fetch (default: 100).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        metavar="SOURCE",
        choices=list(DOLMA_SOURCES),
        default=None,
        help=(
            "One or more Dolma source names. "
            f"Choices: {', '.join(DOLMA_SOURCES)}. "
            "Omit to sample from all sources."
        ),
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=50,
        metavar="CHARS",
        help="Skip documents shorter than this many characters (default: 50).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable URL shuffling (stream shards in list order).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for URL shuffling.",
    )
    parser.add_argument(
        "--url-file",
        default=DOLMA_URL_FILE,
        metavar="HF_PATH",
        help=f"HF Hub path to the URL list (default: {DOLMA_URL_FILE}).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="Hugging Face access token (falls back to HF_TOKEN env var).",
    )

    args = parser.parse_args(argv)

    request = DolmaFetchRequest(
        max_documents=args.max_documents,
        sources=args.sources,
        shuffle_urls=not args.no_shuffle,
        random_seed=args.seed,
        min_text_length=args.min_text_length,
        url_file=args.url_file,
        hf_token=args.hf_token,
    )

    docs = fetch_documents(request)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Fetched {len(docs)} documents -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
