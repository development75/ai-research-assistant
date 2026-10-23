"""Central configuration.

Every tunable number lives here so it can be changed in one place or
overridden with an environment variable (or a line in your .env file).
"""

import os
from pathlib import Path

try:  # python-dotenv is optional at import time so tests can run without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "chat_history.db"

# ---------- Model ----------
APP_NAME = "AI Research Assistant"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
TEMPERATURE = 0
# gpt-oss is a reasoning model: hidden reasoning tokens also count toward this
# limit, so keep it comfortably above the length of a normal answer.
MAX_TOKENS = _int("MAX_TOKENS", 2048)

# ---------- Cost / token control ----------
MAX_HISTORY = _int("MAX_HISTORY", 10)          # past messages sent to the model
MAX_TOOL_CHARS = _int("MAX_TOOL_CHARS", 1500)  # characters of web/wiki text per call
DOC_CONTEXT_CHARS = _int("DOC_CONTEXT_CHARS", 3000)  # characters of document text per call
RECURSION_LIMIT = _int("RECURSION_LIMIT", 15)  # stops runaway tool loops
STREAMING = _bool("STREAMING", True)

# ---------- Web search ----------
WEB_MAX_RESULTS = _int("WEB_MAX_RESULTS", 4)

# ---------- Document Q&A (RAG) ----------
COLLECTION_NAME = "documents"
CHUNK_SIZE = _int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
TOP_K = _int("TOP_K", 4)

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
