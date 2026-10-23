"""Document Q&A: load files, split them into chunks, index them in Chroma, search.

Chroma is imported lazily so the rest of the project (and the unit tests) work
without it. A different collection object can be injected, which is how the
tests run without downloading an embedding model.
"""

import logging
from functools import lru_cache
from pathlib import Path

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOC_CONTEXT_CHARS,
    DOCUMENTS_DIR,
    TOP_K,
)
from .utils import truncate

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
BATCH_SIZE = 64


# ---------- reading and chunking ----------

def read_pages(path: str | Path) -> list[tuple[int, str]]:
    """Return ``[(page_number, text), ...]`` for a PDF, text or markdown file."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
    if extension in {".txt", ".md"}:
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Unsupported file type '{extension or 'none'}'. Supported: {supported}")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring to cut on spaces."""
    if size <= 0 or overlap < 0 or overlap >= size // 2:
        raise ValueError("size must be positive and overlap must be below half of size")
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        space = text.find(" ", start, end)  # begin on a word boundary
        if space != -1:
            start = space + 1
    return chunks


def save_upload(filename: str, data: bytes, directory: str | Path = DOCUMENTS_DIR) -> Path:
    """Save uploaded bytes into the documents folder (path-safe) and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / Path(filename).name  # drop any folder parts
    target.write_bytes(data)
    return target


# ---------- the store ----------

class RagStore:
    def __init__(self, persist_dir: str | Path = CHROMA_DIR, collection=None):
        self.persist_dir = Path(persist_dir)
        self._collection = collection

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(COLLECTION_NAME)
        return self._collection

    def add_document(self, path: str | Path) -> int:
        """Index a file (replacing any older version). Returns the chunk count."""
        path = Path(path)
        records = []
        for page, text in read_pages(path):
            for index, chunk in enumerate(chunk_text(text)):
                records.append(
                    (f"{path.name}::p{page}::c{index}", chunk, {"source": path.name, "page": page})
                )
        if not records:
            raise ValueError(
                f"No readable text found in {path.name}. Scanned PDFs need OCR first."
            )

        self.delete_document(path.name)
        collection = self.collection
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            collection.upsert(
                ids=[item[0] for item in batch],
                documents=[item[1] for item in batch],
                metadatas=[item[2] for item in batch],
            )
        logger.info("indexed %s: %d chunks", path.name, len(records))
        return len(records)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        collection = self.collection
        total = collection.count()
        if total == 0:
            return []
        result = collection.query(query_texts=[query], n_results=min(k, total))
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = (result.get("distances") or [[None] * len(documents)])[0]
        return [
            {
                "text": text,
                "source": (meta or {}).get("source", "unknown"),
                "page": (meta or {}).get("page"),
                "distance": distance,
            }
            for text, meta, distance in zip(documents, metadatas, distances)
        ]

    def list_documents(self) -> dict[str, int]:
        """``{file name: number of chunks}`` for everything indexed."""
        data = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            source = (meta or {}).get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def delete_document(self, name: str) -> None:
        self.collection.delete(where={"source": name})


# ---------- text for the model ----------

def format_results(results: list[dict], limit: int = DOC_CONTEXT_CHARS) -> str:
    """Numbered passages with file name and page, split into equal budgets."""
    if not results:
        return ""
    per_result = max(limit // len(results) - 60, 100)
    blocks = []
    for number, item in enumerate(results, start=1):
        page = f", page {item['page']}" if item.get("page") else ""
        blocks.append(f"[{number}] {item['source']}{page}\n{truncate(item['text'], per_result)}")
    return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def get_store() -> RagStore:
    return RagStore()


def search_documents_text(
    query: str, store: RagStore | None = None, k: int = TOP_K, limit: int = DOC_CONTEXT_CHARS
) -> str:
    query = (query or "").strip()
    if not query:
        return "Document search failed: the query was empty."
    try:
        results = (store or get_store()).search(query, k)
    except Exception as exc:
        logger.warning("document search failed for %r: %s", query, exc)
        return f"Document search failed: {exc}"
    if not results:
        return "No matching content found. The user may not have uploaded any documents yet."
    return format_results(results, limit)
