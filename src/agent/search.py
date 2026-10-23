"""Web and Wikipedia search logic (kept free of LangChain so it is testable).

Both functions return plain text that always includes the source URLs, so the
model can cite them. Output is size-limited to keep token usage low.
"""

import logging
from urllib.parse import quote

import requests

from .config import MAX_TOOL_CHARS, WEB_MAX_RESULTS
from .utils import truncate

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
HEADERS = {"User-Agent": "AI-Research-Assistant/1.0 (portfolio project)"}
TIMEOUT = 10


# ---------- Web search (DuckDuckGo) ----------

def _ddg_text(query: str, max_results: int) -> list[dict]:
    """Call DuckDuckGo. Isolated in its own function so tests can replace it."""
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    return list(DDGS().text(query, max_results=max_results))


def search_web(
    query: str, max_results: int = WEB_MAX_RESULTS, limit: int = MAX_TOOL_CHARS
) -> str:
    query = (query or "").strip()
    if not query:
        return "Web search failed: the query was empty."
    try:
        results = _ddg_text(query, max_results)
    except Exception as exc:  # network errors, rate limits, missing package
        logger.warning("web_search failed for %r: %s", query, exc)
        return f"Web search failed: {exc}"
    if not results:
        return "No web results found."

    # Give every result its own budget so every URL survives truncation.
    per_result = max(limit // len(results) - 120, 80)
    blocks = []
    for number, item in enumerate(results, start=1):
        title = " ".join(str(item.get("title", "")).split())
        body = " ".join(str(item.get("body", "")).split())
        url = item.get("href") or item.get("url") or ""
        blocks.append(f"[{number}] {title}\n{truncate(body, per_result)}\nURL: {url}")
    return "\n\n".join(blocks)


# ---------- Wikipedia ----------

def _wiki_find_title(query: str) -> str | None:
    response = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        },
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    hits = response.json().get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def _wiki_summary(title: str) -> dict:
    url = WIKI_SUMMARY.format(title=quote(title.replace(" ", "_"), safe=""))
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    page_url = (
        data.get("content_urls", {}).get("desktop", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    )
    return {
        "title": data.get("title", title),
        "extract": data.get("extract", ""),
        "url": page_url,
    }


def wikipedia_text(query: str, limit: int = MAX_TOOL_CHARS) -> str:
    query = (query or "").strip()
    if not query:
        return "Wikipedia search failed: the query was empty."
    try:
        title = _wiki_find_title(query)
        if not title:
            return f'No Wikipedia article found for "{query}".'
        page = _wiki_summary(title)
    except Exception as exc:
        logger.warning("wikipedia_search failed for %r: %s", query, exc)
        return f"Wikipedia search failed: {exc}"

    header = f"Wikipedia: {page['title']}\n"
    footer = f"\nURL: {page['url']}"
    room = max(limit - len(header) - len(footer), 100)
    return header + truncate(page["extract"], room) + footer
