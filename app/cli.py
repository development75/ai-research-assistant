"""Terminal interface for the AI Research Assistant.

Run from the project root:  python -m app.cli
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import config  # noqa: E402  (also loads the .env file)
from src.agent.agent import (  # noqa: E402
    MissingApiKeyError,
    build_agent,
    run_reply,
    stream_events,
)
from src.agent.memory import ChatStore  # noqa: E402
from src.agent.rag import SUPPORTED_EXTENSIONS, get_store  # noqa: E402
from src.agent.utils import setup_logging  # noqa: E402

HELP = """Commands:
  /help            show this help
  /new             start a new conversation
  /history         list saved conversations
  /load <id>       continue a saved conversation
  /docs            list indexed documents
  /add <path>      index a PDF, .txt or .md file for document Q&A
  /remove <name>   remove an indexed document
  exit | quit      leave
"""


def show_conversation(store: ChatStore, conversation_id: int) -> None:
    for message in store.get_messages(conversation_id):
        label = "You" if message["role"] == "user" else "Agent"
        print(f"{label}: {message['content']}\n")


def handle_command(line: str, store: ChatStore, conversation_id: int | None):
    """Run a slash command. Returns the (possibly new) conversation id."""
    command, _, argument = line.partition(" ")
    argument = argument.strip()

    if command == "/help":
        print(HELP)
    elif command == "/new":
        print("Started a new conversation.\n")
        return None
    elif command == "/history":
        conversations = store.list_conversations(20)
        if not conversations:
            print("No saved conversations yet.\n")
        for item in conversations:
            print(f"  {item['id']:>3}  {item['title']}  ({item['updated_at']})")
        print()
    elif command == "/load":
        if not argument.isdigit() or store.get_conversation(int(argument)) is None:
            print("Usage: /load <id>  (see /history for ids)\n")
        else:
            conversation_id = int(argument)
            print()
            show_conversation(store, conversation_id)
    elif command == "/docs":
        documents = get_store().list_documents()
        if not documents:
            print("No documents indexed yet. Use /add <path>.\n")
        for name, chunks in documents.items():
            print(f"  {name}  ({chunks} chunks)")
        print()
    elif command == "/add":
        path = Path(argument.strip('"'))
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}\n")
        elif not path.is_file():
            print(f"File not found: {path}\n")
        else:
            try:
                count = get_store().add_document(path)
                print(f"Indexed {path.name} ({count} chunks).\n")
            except Exception as exc:
                print(f"Could not index that file: {exc}\n")
    elif command == "/remove":
        get_store().delete_document(argument)
        print(f"Removed {argument}.\n")
    else:
        print("Unknown command. Type /help.\n")
    return conversation_id


def answer(agent, messages: list[dict], stream: bool):
    """Print the agent's reply. Returns (text, tools_used)."""
    if not stream:
        reply = run_reply(agent, messages)
        for name in reply.tools_used:
            print(f"  [tool] {name}")
        print(f"Agent: {reply.text}\n")
        return reply.text, reply.tools_used

    parts: list[str] = []
    tools: list[str] = []
    started = False  # True while an "Agent:" line is being written
    for event in stream_events(agent, messages):
        if event.kind == "tool":
            if event.value not in tools:
                tools.append(event.value)
            if started:
                print()
                started = False
            print(f"  [tool] {event.value}", flush=True)
        else:
            if not started:
                print("Agent: ", end="", flush=True)
                started = True
            parts.append(event.value)
            print(event.value, end="", flush=True)
    print("\n")
    return "".join(parts), tools


def main() -> None:
    # Windows consoles often use a legacy encoding; avoid crashes on symbols.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=config.APP_NAME)
    parser.add_argument("--no-stream", action="store_true", help="wait for the full answer")
    parser.add_argument("--resume", type=int, metavar="ID", help="continue a saved conversation")
    args = parser.parse_args()

    setup_logging()
    try:
        agent = build_agent()
    except MissingApiKeyError as exc:
        sys.exit(str(exc))

    store = ChatStore()
    conversation_id = args.resume if args.resume and store.get_conversation(args.resume) else None
    stream = config.STREAMING and not args.no_stream

    print(f"{config.APP_NAME}. Type /help for commands, 'exit' to quit.\n")
    if conversation_id:
        show_conversation(store, conversation_id)

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            print("Agent: Goodbye! Have a great day.")
            break
        if line.startswith("/"):
            try:
                conversation_id = handle_command(line, store, conversation_id)
            except Exception as exc:  # e.g. document index not available
                print(f"Command failed: {exc}\n")
            continue

        history = store.history(conversation_id) if conversation_id else []
        messages = history + [{"role": "user", "content": line}]
        try:
            text, tools = answer(agent, messages, stream)
        except Exception as exc:  # keep the chat alive on API/network errors
            print(f"\nAgent: Sorry, something went wrong: {exc}\n")
            continue

        if conversation_id is None:
            conversation_id = store.create_conversation()
        store.add_message(conversation_id, "user", line)
        store.add_message(conversation_id, "assistant", text, tools)


if __name__ == "__main__":
    main()
