"""LangChain tools the agent can call.

Each tool is a thin wrapper: the real logic lives in calc.py, search.py and
rag.py, where it can be unit-tested without LangChain.
"""

import logging
import time
from typing import Literal

from langchain_core.tools import tool

from . import calc, rag, search
from .utils import current_time_text

logger = logging.getLogger(__name__)


def _log(name: str, detail: str, started: float, output: str) -> None:
    logger.info(
        "tool=%s input=%r chars=%d seconds=%.2f",
        name, detail, len(output), time.perf_counter() - started,
    )


@tool
def get_current_time() -> str:
    """Get the current date and time. Use this when the user asks about today's date or the time."""
    return current_time_text()


@tool
def calculate(
    a: float,
    b: float,
    operation: Literal["add", "subtract", "multiply", "divide", "power"],
) -> str:
    """Do arithmetic on two numbers. Always use this instead of calculating yourself.

    Args:
        a: The first number.
        b: The second number.
        operation: One of add, subtract, multiply, divide, power.
    """
    try:
        return calc.format_number(calc.compute(a, b, operation))
    except ValueError as exc:
        return f"Error: {exc}"


@tool
def web_search(query: str) -> str:
    """Search the internet for current or recent information such as news, prices,
    weather, sports, or anything that may have changed recently. Input is a short
    search query. Returns numbered results with their URLs."""
    started = time.perf_counter()
    output = search.search_web(query)
    _log("web_search", query, started, output)
    return output


@tool
def wikipedia_search(query: str) -> str:
    """Look up background information on people, places, history, science and
    concepts from Wikipedia. Input is the topic name or a short query. Returns a
    summary and the article URL."""
    started = time.perf_counter()
    output = search.wikipedia_text(query)
    _log("wikipedia_search", query, started, output)
    return output


@tool
def search_documents(query: str) -> str:
    """Search the user's own uploaded documents (PDFs, text and markdown notes).
    Use this when the question may be answered by files the user has uploaded.
    Input is a natural-language search query. Returns passages with file name and page."""
    started = time.perf_counter()
    output = rag.search_documents_text(query)
    _log("search_documents", query, started, output)
    return output


ALL_TOOLS = [get_current_time, calculate, web_search, wikipedia_search, search_documents]
