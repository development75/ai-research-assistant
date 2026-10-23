import unittest
from datetime import datetime, timezone

from src.agent.utils import TRUNCATION_MARKER, current_time_text, extract_text, truncate


class TruncateTests(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(truncate("hello", 10), "hello")

    def test_long_text_is_cut_with_marker(self):
        result = truncate("a" * 100, 20)
        self.assertTrue(result.endswith(TRUNCATION_MARKER))
        self.assertLessEqual(len(result), 20 + len(TRUNCATION_MARKER))

    def test_exact_limit_is_unchanged(self):
        self.assertEqual(truncate("a" * 20, 20), "a" * 20)

    def test_non_string_input(self):
        self.assertEqual(truncate(12345, 3), "123" + TRUNCATION_MARKER)


class ExtractTextTests(unittest.TestCase):
    def test_string(self):
        self.assertEqual(extract_text("hi"), "hi")

    def test_none(self):
        self.assertEqual(extract_text(None), "")

    def test_content_blocks(self):
        blocks = [{"type": "text", "text": "Hello "}, {"type": "image", "url": "x"}, "world"]
        self.assertEqual(extract_text(blocks), "Hello world")


class CurrentTimeTests(unittest.TestCase):
    def test_format(self):
        moment = datetime(2026, 10, 23, 14, 5, tzinfo=timezone.utc)
        text = current_time_text(moment)
        self.assertIn("Friday, 23 October 2026", text)
        self.assertIn("02:05 PM", text)


if __name__ == "__main__":
    unittest.main()
