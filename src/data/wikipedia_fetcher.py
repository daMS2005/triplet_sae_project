"""Utilities for collecting Wikipedia article text via MediaWiki API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class WikipediaFetchRequest:
    language: str = "en"
    max_articles: int = 100
    titles: list[str] | None = None
    category: str | None = None
    search_query: str | None = None
    random_articles: bool = False
    random_categories: bool = False
    category_depth: int = 0
    random_category_pool: int = 5
    random_seed: int | None = None
    sleep_seconds: float = 0.1
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_articles <= 0:
            raise ValueError("max_articles must be > 0")
        if self.category_depth < 0:
            raise ValueError("category_depth must be >= 0")
        if self.random_category_pool <= 0:
            raise ValueError("random_category_pool must be > 0")

        sources = [
            bool(self.titles),
            bool(self.category),
            bool(self.search_query),
            self.random_articles,
            self.random_categories,
        ]
        if sum(sources) != 1:
            raise ValueError(
                "Provide exactly one source: titles, category, search_query, "
                "random_articles, or random_categories."
            )


def _default_user_agent() -> str:
    """
    Wikimedia requests a descriptive User-Agent with contact info.
    Allow override via env var WIKIPEDIA_USER_AGENT.
    """
    ua = os.environ.get("WIKIPEDIA_USER_AGENT")
    if ua and ua.strip():
        return ua.strip()
    return "triplet_sae_project/0.1 (contact: you@example.com)"


class MediaWikiClient:
    def __init__(
        self,
        language: str = "en",
        timeout_seconds: float = 30.0,
        sleep_seconds: float = 0.0,
        user_agent: str | None = None,
    ) -> None:
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.timeout_seconds = timeout_seconds
        self.sleep_seconds = sleep_seconds
        self._last_request_at = 0.0

        ua = (user_agent or _default_user_agent()).strip()
        self._headers = {
            "User-Agent": ua,
            "Accept": "application/json",
        }

    def _request(self, params: dict[str, str]) -> dict:
        if self.sleep_seconds > 0:
            now = time.monotonic()
            elapsed = now - self._last_request_at
            remaining = self.sleep_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)

        query = urlencode(params)
        url = f"{self.api_url}?{query}"
        try:
            req = Request(url, headers=self._headers, method="GET")
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                payload = resp.read().decode("utf-8")
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"MediaWiki request failed: {url}") from exc
        self._last_request_at = time.monotonic()
        return json.loads(payload)

    def search_titles(self, query: str, limit: int) -> list[str]:
        titles: list[str] = []
        offset = 0
        while len(titles) < limit:
            batch = min(50, limit - len(titles))
            data = self._request(
                {
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": str(batch),
                    "sroffset": str(offset),
                }
            )
            items = data.get("query", {}).get("search", [])
            if not items:
                break
            titles.extend(item["title"] for item in items if "title" in item)
            offset += len(items)
        return _dedupe(titles)

    def category_titles(self, category_name: str, limit: int, depth: int) -> list[str]:
        visited_categories: set[str] = set()
        out_titles: list[str] = []

        def walk(cat: str, remaining_depth: int) -> None:
            if len(out_titles) >= limit:
                return
            full_cat = cat if cat.startswith("Category:") else f"Category:{cat}"
            if full_cat in visited_categories:
                return
            visited_categories.add(full_cat)

            continue_token = ""
            while True:
                cm_limit = min(100, limit - len(out_titles))
                if cm_limit <= 0:
                    return
                params = {
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmtitle": full_cat,
                    "cmlimit": str(cm_limit),
                }
                if continue_token:
                    params["cmcontinue"] = continue_token
                data = self._request(params)
                members = data.get("query", {}).get("categorymembers", [])
                for member in members:
                    title = member.get("title", "")
                    ns = member.get("ns", -1)
                    if ns == 0 and title:
                        out_titles.append(title)
                        if len(out_titles) >= limit:
                            return
                    elif ns == 14 and remaining_depth > 0 and title:
                        walk(title, remaining_depth - 1)
                        if len(out_titles) >= limit:
                            return

                continue_token = data.get("continue", {}).get("cmcontinue", "")
                if not continue_token:
                    break

        walk(category_name, depth)
        return _dedupe(out_titles)[:limit]

    def get_article_extracts(self, titles: Iterable[str]) -> list[dict]:
        titles = [t for t in titles if t]
        if not titles:
            return []

        # Chunk titles to keep query size bounded; MediaWiki title separator is '|'.
        title_chunks: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for title in titles:
            next_len = current_len + len(title) + (1 if current else 0)
            if next_len > 1500 and current:
                title_chunks.append(current)
                current = [title]
                current_len = len(title)
            else:
                current.append(title)
                current_len = next_len
        if current:
            title_chunks.append(current)

        results: list[dict] = []
        fetched_at = datetime.now(timezone.utc).isoformat()

        def _append_from_page(page: dict) -> None:
            title = page.get("title")
            extract = page.get("extract", "")
            if not title or not extract:
                return
            revisions = page.get("revisions", [])
            rev0 = revisions[0] if revisions else {}
            results.append(
                {
                    "article_id": str(page.get("pageid", "")),
                    "title": title,
                    "source_url": page.get(
                        "fullurl",
                        f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    ),
                    "revision_id": str(rev0.get("revid", "")),
                    "revision_timestamp": rev0.get("timestamp", ""),
                    "text": _clean_text(extract),
                    "retrieved_at": fetched_at,
                }
            )

        for chunk in title_chunks:
            if len(results) >= len(titles):
                break

            data = self._request(
                {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts|revisions|info",
                    "titles": "|".join(chunk),
                    "explaintext": "1",
                    "redirects": "1",
                    "rvprop": "ids|timestamp",
                    "inprop": "url",
                }
            )

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                title = page.get("title")
                extract = page.get("extract", "")

                # Fallback: if extract missing/empty, fetch this title alone
                if title and not extract:
                    single = self._request(
                        {
                            "action": "query",
                            "format": "json",
                            "prop": "extracts|revisions|info",
                            "titles": title,
                            "explaintext": "1",
                            "redirects": "1",
                            "rvprop": "ids|timestamp",
                            "inprop": "url",
                        }
                    )
                    spages = single.get("query", {}).get("pages", {})
                    spage = next(iter(spages.values()), {})
                    if spage:
                        page = spage

                _append_from_page(page)

        return results

    def random_titles(self, limit: int) -> list[str]:
        titles: list[str] = []
        while len(titles) < limit:
            batch = min(50, limit - len(titles))
            data = self._request(
                {
                    "action": "query",
                    "format": "json",
                    "list": "random",
                    "rnnamespace": "0",
                    "rnlimit": str(batch),
                }
            )
            items = data.get("query", {}).get("random", [])
            if not items:
                break
            titles.extend(item["title"] for item in items if "title" in item)
        return _dedupe(titles)[:limit]

    def random_category_titles(self, limit: int) -> list[str]:
        categories: list[str] = []
        while len(categories) < limit:
            batch = min(50, limit - len(categories))
            data = self._request(
                {
                    "action": "query",
                    "format": "json",
                    "list": "random",
                    "rnnamespace": "14",
                    "rnlimit": str(batch),
                }
            )
            items = data.get("query", {}).get("random", [])
            if not items:
                break
            categories.extend(item["title"] for item in items if "title" in item)
        return _dedupe(categories)[:limit]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def read_titles_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    titles = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return _dedupe(titles)


def fetch_articles(request: WikipediaFetchRequest) -> list[dict]:
    client = MediaWikiClient(
        language=request.language,
        timeout_seconds=request.timeout_seconds,
        sleep_seconds=request.sleep_seconds,
        # User-Agent can be set via env var WIKIPEDIA_USER_AGENT (see _default_user_agent)
    )

    if request.titles:
        titles = request.titles[: request.max_articles]
    elif request.category:
        titles = client.category_titles(
            category_name=request.category,
            limit=request.max_articles,
            depth=request.category_depth,
        )
    elif request.search_query:
        titles = client.search_titles(
            query=request.search_query,
            limit=request.max_articles,
        )
    elif request.random_articles:
        titles = client.random_titles(request.max_articles)
    elif request.random_categories:
        rng = random.Random(request.random_seed)
        titles = []
        target_pool = request.random_category_pool
        attempts = 0
        max_attempts = max(3, target_pool * 3)

        while len(titles) < request.max_articles and attempts < max_attempts:
            attempts += 1
            categories = client.random_category_titles(target_pool)
            if not categories:
                break
            rng.shuffle(categories)
            for category in categories:
                if len(titles) >= request.max_articles:
                    break
                needed = request.max_articles - len(titles)
                cat_titles = client.category_titles(
                    category_name=category,
                    limit=needed,
                    depth=request.category_depth,
                )
                rng.shuffle(cat_titles)
                titles.extend(cat_titles)
            titles = _dedupe(titles)
    else:
        raise ValueError(
            "Provide exactly one source: titles, category, search_query, "
            "random_articles, or random_categories."
        )

    articles = client.get_article_extracts(titles)
    return articles[: request.max_articles]