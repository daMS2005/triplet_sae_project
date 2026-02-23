#!/usr/bin/env python3
"""Fetch Wikipedia articles and store them as JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wikipedia_fetcher import (  # noqa: E402
    WikipediaFetchRequest,
    fetch_articles,
    read_titles_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia articles and save records as JSONL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/wikipedia_articles.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Wikipedia language edition, e.g. en, es, de.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=100,
        help="Maximum number of articles to fetch.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help="Delay between requests to avoid hammering API.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        help=(
            "Optional User-Agent string for Wikimedia API. "
            "If provided, will be exported as WIKIPEDIA_USER_AGENT for the fetcher to use."
        ),
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--titles-file",
        type=Path,
        help="Path to newline-separated article titles.",
    )
    source.add_argument(
        "--category",
        help="Wikipedia category name, e.g. Physics or Machine_learning.",
    )
    source.add_argument(
        "--search-query",
        help="Search query for discovering article titles.",
    )
    source.add_argument(
        "--random-articles",
        action="store_true",
        help="Fetch random articles directly from Wikipedia.",
    )
    source.add_argument(
        "--random-categories",
        action="store_true",
        help="Pick random categories, then sample articles from them.",
    )

    parser.add_argument(
        "--category-depth",
        type=int,
        default=0,
        help="Depth for category traversal (0 = category only).",
    )
    parser.add_argument(
        "--random-category-pool",
        type=int,
        default=5,
        help="How many random categories to sample per pass in random-category mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible shuffling in random modes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Allow passing a UA via CLI; fetcher can read this env var if implemented
    if args.user_agent:
        os.environ["WIKIPEDIA_USER_AGENT"] = args.user_agent

    titles = None
    if args.titles_file:
        titles = read_titles_file(args.titles_file)
        if not titles:
            raise ValueError(f"No titles found in file: {args.titles_file}")

    request = WikipediaFetchRequest(
        language=args.language,
        max_articles=args.max_articles,
        titles=titles,
        category=args.category,
        search_query=args.search_query,
        random_articles=args.random_articles,
        random_categories=args.random_categories,
        category_depth=args.category_depth,
        random_category_pool=args.random_category_pool,
        random_seed=args.seed,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
    )

    articles = fetch_articles(request)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as out_f:
        for article in articles:
            out_f.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"Fetched {len(articles)} articles -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())