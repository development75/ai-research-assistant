"""Small shared helpers."""

import logging
from datetime import datetime

from .config import LOG_LEVEL, MAX_TOOL_CHARS

TRUNCATION_MARKER = "... [truncated]"


def truncate(text, limit: int = MAX_TOOL_CHARS) -> str:
    """Cut ``text`` to at most ``limit`` characters (plus a short marker)."""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: max(limit, 0)].rstrip() + TRUNCATION_MARKER


def extract_text(content) -> str:
    """Return plain text from a model message ``content``.

    Some providers return a string, others a list of content blocks.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def current_time_text(now: datetime | None = None) -> str:
    """Human readable local date and time, including the timezone."""
    now = now or datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %I:%M %p %Z").strip()


def setup_logging(level: str | None = None) -> None:
    """Configure logging once for the whole app."""
    logging.basicConfig(
        level=getattr(logging, (level or LOG_LEVEL).upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
