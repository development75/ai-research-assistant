"""Builds the agent and turns its raw output into simple events.

LangChain and Groq are imported lazily inside ``build_agent`` so the pure
helpers below can be tested without those packages installed.
"""

import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator, NamedTuple

from .config import MAX_TOKENS, MODEL_NAME, RECURSION_LIMIT, TEMPERATURE
from .prompts import SYSTEM_PROMPT
from .utils import extract_text

RUN_CONFIG = {"recursion_limit": RECURSION_LIMIT}


class MissingApiKeyError(RuntimeError):
    """Raised when GROQ_API_KEY is not set."""


class Event(NamedTuple):
    kind: str   # "token" (piece of the answer) or "tool" (a tool was used)
    value: str


@dataclass
class Reply:
    text: str
    tools_used: list[str] = field(default_factory=list)


def build_agent():
    """Create the LangChain agent (model + tools + system prompt)."""
    if not os.getenv("GROQ_API_KEY"):
        raise MissingApiKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/"
        )
    from langchain.agents import create_agent
    from langchain_groq import ChatGroq

    from .tools import ALL_TOOLS

    llm = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    return create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


# ---------- non-streaming ----------

def tools_used_in_turn(messages) -> list[str]:
    """Names of tools called, in order, without duplicates."""
    names: list[str] = []
    for message in messages:
        if getattr(message, "type", None) == "tool":
            name = getattr(message, "name", None)
            if name and name not in names:
                names.append(name)
    return names


def run_reply(agent, messages: list[dict]) -> Reply:
    """Ask the agent and wait for the complete answer."""
    result = agent.invoke({"messages": messages}, config=RUN_CONFIG)
    all_messages = result["messages"]
    return Reply(
        text=extract_text(all_messages[-1].content),
        tools_used=tools_used_in_turn(all_messages),
    )


# ---------- streaming ----------

def events_from_chunks(raw: Iterable) -> Iterator[Event]:
    """Convert LangGraph ``stream_mode="messages"`` output into Events.

    Accepts ``(chunk, metadata)`` tuples or bare chunks. Works by duck typing,
    so it can be tested with simple fake objects.
    """
    seen_calls: set = set()
    streamed_any = False
    last_full_text = ""

    for item in raw:
        chunk = item[0] if isinstance(item, tuple) else item
        kind = getattr(chunk, "type", None)

        if kind == "tool":  # a tool finished
            key = getattr(chunk, "tool_call_id", None) or id(chunk)
            name = getattr(chunk, "name", None)
            if name and key not in seen_calls:
                seen_calls.add(key)
                yield Event("tool", name)
            continue

        if kind not in ("AIMessageChunk", "ai"):  # ignore human/system echoes
            continue

        for call in getattr(chunk, "tool_call_chunks", None) or []:
            name, key = call.get("name"), call.get("id")
            if name and key and key not in seen_calls:
                seen_calls.add(key)
                yield Event("tool", name)

        text = extract_text(getattr(chunk, "content", ""))
        if not text:
            continue
        if kind == "AIMessageChunk":
            streamed_any = True
            yield Event("token", text)
        else:  # a complete message; only used if nothing was streamed
            last_full_text = text

    if not streamed_any and last_full_text:
        yield Event("token", last_full_text)


def stream_events(agent, messages: list[dict]) -> Iterator[Event]:
    """Ask the agent and yield Events as they happen."""
    raw = agent.stream({"messages": messages}, config=RUN_CONFIG, stream_mode="messages")
    yield from events_from_chunks(raw)
