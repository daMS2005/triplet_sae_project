import pytest

from src.data.wikipedia_fetcher import WikipediaFetchRequest, _clean_text


def test_clean_text_collapses_whitespace() -> None:
    raw = "Hello\t\tworld\n\n\nThis   is  text.\xa0"
    assert _clean_text(raw) == "Hello world\n\nThis is text."


def test_request_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        WikipediaFetchRequest(
            search_query="graph",
            random_articles=True,
        )
