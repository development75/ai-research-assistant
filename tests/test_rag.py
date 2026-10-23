import re
import tempfile
import unittest
from pathlib import Path

from src.agent.rag import (
    RagStore,
    chunk_text,
    format_results,
    read_pages,
    save_upload,
    search_documents_text,
)


class FakeCollection:
    """Tiny in-memory stand-in for a Chroma collection (word-overlap ranking)."""

    def __init__(self):
        self.rows = {}  # id -> (document, metadata)

    def count(self):
        return len(self.rows)

    def upsert(self, ids, documents, metadatas):
        for item_id, doc, meta in zip(ids, documents, metadatas):
            self.rows[item_id] = (doc, meta)

    def delete(self, where):
        key, value = next(iter(where.items()))
        self.rows = {i: (d, m) for i, (d, m) in self.rows.items() if m.get(key) != value}

    def get(self, include=None):
        return {"ids": list(self.rows), "metadatas": [m for _, m in self.rows.values()]}

    def query(self, query_texts, n_results):
        words = set(re.findall(r"\w+", query_texts[0].lower()))
        scored = sorted(
            self.rows.values(),
            key=lambda row: -len(words & set(re.findall(r"\w+", row[0].lower()))),
        )[:n_results]
        return {
            "documents": [[doc for doc, _ in scored]],
            "metadatas": [[meta for _, meta in scored]],
            "distances": [[0.1 * i for i in range(len(scored))]],
        }


class ChunkTextTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunk_text("   \n "), [])

    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("hello   world", size=100, overlap=10), ["hello world"])

    def test_chunks_respect_size_and_cover_everything(self):
        words = [f"word{i}" for i in range(300)]
        text = " ".join(words)
        chunks = chunk_text(text, size=200, overlap=40)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        joined = " ".join(chunks)
        for word in words:
            self.assertIn(word, joined)

    def test_chunks_overlap(self):
        text = " ".join(f"w{i}" for i in range(200))
        chunks = chunk_text(text, size=100, overlap=30)
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        self.assertTrue(first_words & second_words)

    def test_very_long_word_still_terminates(self):
        chunks = chunk_text("x" * 1000, size=100, overlap=10)
        self.assertGreater(len(chunks), 1)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", size=100, overlap=60)
        with self.assertRaises(ValueError):
            chunk_text("abc", size=0, overlap=0)


class FileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_read_text_file(self):
        path = self.dir / "notes.txt"
        path.write_text("Some notes", encoding="utf-8")
        self.assertEqual(read_pages(path), [(1, "Some notes")])

    def test_unsupported_type(self):
        path = self.dir / "image.png"
        path.write_bytes(b"x")
        with self.assertRaises(ValueError):
            read_pages(path)

    def test_save_upload_strips_folders(self):
        saved = save_upload("../../evil.txt", b"data", self.dir / "docs")
        self.assertEqual(saved.parent, self.dir / "docs")
        self.assertEqual(saved.name, "evil.txt")
        self.assertEqual(saved.read_bytes(), b"data")


class RagStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.store = RagStore(self.dir / "chroma", collection=FakeCollection())

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_add_search_and_cite(self):
        self.store.add_document(self.write("solar.md", "Solar panels convert sunlight into electricity."))
        self.store.add_document(self.write("cooking.txt", "Boil pasta for ten minutes in salted water."))
        results = self.store.search("how do solar panels work", k=1)
        self.assertEqual(results[0]["source"], "solar.md")
        self.assertEqual(results[0]["page"], 1)

    def test_list_and_delete(self):
        self.store.add_document(self.write("a.txt", "alpha text here"))
        self.store.add_document(self.write("b.txt", "beta text here"))
        self.assertEqual(list(self.store.list_documents()), ["a.txt", "b.txt"])
        self.store.delete_document("a.txt")
        self.assertEqual(list(self.store.list_documents()), ["b.txt"])

    def test_reindexing_replaces_old_chunks(self):
        path = self.write("report.txt", "old content " * 200)
        first = self.store.add_document(path)
        self.assertGreater(first, 1)
        path.write_text("new short content", encoding="utf-8")
        self.store.add_document(path)
        self.assertEqual(self.store.list_documents(), {"report.txt": 1})

    def test_empty_file_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add_document(self.write("empty.txt", "   "))

    def test_search_on_empty_store(self):
        self.assertEqual(self.store.search("anything"), [])
        self.assertIn("No matching content", search_documents_text("anything", store=self.store))


class FormatResultsTests(unittest.TestCase):
    def test_includes_source_and_page_for_every_result(self):
        results = [
            {"text": "x" * 2000, "source": "a.pdf", "page": 3},
            {"text": "y" * 2000, "source": "b.pdf", "page": 7},
        ]
        output = format_results(results, limit=600)
        self.assertIn("a.pdf, page 3", output)
        self.assertIn("b.pdf, page 7", output)
        self.assertLess(len(output), 900)

    def test_empty(self):
        self.assertEqual(format_results([]), "")


class SearchTextErrorTests(unittest.TestCase):
    def test_store_errors_become_text(self):
        class Broken:
            def search(self, query, k):
                raise RuntimeError("index offline")

        self.assertIn("index offline", search_documents_text("hello", store=Broken()))

    def test_empty_query(self):
        self.assertIn("empty", search_documents_text("  "))


if __name__ == "__main__":
    unittest.main()
