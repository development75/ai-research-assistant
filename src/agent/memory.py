"""Persistent chat history stored in SQLite (standard library only)."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH, MAX_HISTORY

DEFAULT_TITLE = "New chat"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    tools_used      TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_title(text: str, max_length: int = 50) -> str:
    text = " ".join(text.split())
    if len(text) <= max_length:
        return text or DEFAULT_TITLE
    return text[: max_length - 1].rstrip() + "…"


class ChatStore:
    """Conversations and messages, saved between runs."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------- conversations ----------

    def create_conversation(self, title: str = DEFAULT_TITLE) -> int:
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            return cursor.lastrowid

    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (make_title(title), conversation_id),
            )

    def delete_conversation(self, conversation_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    # ---------- messages ----------

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        tools_used: list[str] | None = None,
    ) -> int:
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tools_used, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, json.dumps(tools_used or []), now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            if role == "user":  # the first user message names the chat
                conn.execute(
                    "UPDATE conversations SET title = ? WHERE id = ? AND title = ?",
                    (make_title(content), conversation_id, DEFAULT_TITLE),
                )
            return cursor.lastrowid

    def get_messages(self, conversation_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, tools_used, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "tools_used": json.loads(row["tools_used"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def history(self, conversation_id: int, limit: int = MAX_HISTORY) -> list[dict]:
        """The last ``limit`` messages as plain ``{"role", "content"}`` dicts.

        Ready to send to the model. Leading assistant messages are dropped so
        the window always starts with a user turn.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        messages = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        return messages
