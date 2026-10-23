import unittest
from unittest import mock

from src.agent import search


class WebSearchTests(unittest.TestCase):
    def test_formats_results_with_urls(self):
        fake = [
            {"title": "ISRO launch", "body": "A rocket launched today.", "href": "https://example.com/a"},
            {"title": "Space news", "body": "More news.", "href": "https://example.com/b"},
        ]
        with mock.patch.object(search, "_ddg_text", return_value=fake):
            output = search.search_web("isro news")
        self.assertIn("[1] ISRO launch", output)
        self.assertIn("URL: https://example.com/a", output)
        self.assertIn("URL: https://example.com/b", output)

    def test_every_url_survives_long_snippets(self):
        fake = [
            {"title": f"T{i}", "body": "word " * 500, "href": f"https://example.com/{i}"}
            for i in range(4)
        ]
        with mock.patch.object(search, "_ddg_text", return_value=fake):
            output = search.search_web("q", limit=1500)
        for i in range(4):
            self.assertIn(f"https://example.com/{i}", output)
        self.assertLess(len(output), 1800)

    def test_no_results(self):
        with mock.patch.object(search, "_ddg_text", return_value=[]):
            self.assertEqual(search.search_web("q"), "No web results found.")

    def test_errors_become_text_instead_of_raising(self):
        with mock.patch.object(search, "_ddg_text", side_effect=RuntimeError("rate limited")):
            output = search.search_web("q")
        self.assertIn("Web search failed", output)
        self.assertIn("rate limited", output)

    def test_empty_query_does_not_call_the_network(self):
        with mock.patch.object(search, "_ddg_text") as fake:
            self.assertIn("empty", search.search_web("   "))
        fake.assert_not_called()


class WikipediaTests(unittest.TestCase):
    def test_summary_with_url(self):
        page = {"title": "Nikola Tesla", "extract": "Serbian-American inventor.", "url": "https://en.wikipedia.org/wiki/Nikola_Tesla"}
        with mock.patch.object(search, "_wiki_find_title", return_value="Nikola Tesla"), \
                mock.patch.object(search, "_wiki_summary", return_value=page):
            output = search.wikipedia_text("tesla")
        self.assertIn("Wikipedia: Nikola Tesla", output)
        self.assertIn("Serbian-American inventor.", output)
        self.assertIn("URL: https://en.wikipedia.org/wiki/Nikola_Tesla", output)

    def test_long_extract_is_truncated_but_url_kept(self):
        page = {"title": "Long", "extract": "word " * 2000, "url": "https://en.wikipedia.org/wiki/Long"}
        with mock.patch.object(search, "_wiki_find_title", return_value="Long"), \
                mock.patch.object(search, "_wiki_summary", return_value=page):
            output = search.wikipedia_text("long", limit=500)
        self.assertLess(len(output), 700)
        self.assertTrue(output.endswith("https://en.wikipedia.org/wiki/Long"))

    def test_article_not_found(self):
        with mock.patch.object(search, "_wiki_find_title", return_value=None):
            self.assertIn("No Wikipedia article found", search.wikipedia_text("zzzz"))

    def test_network_error(self):
        with mock.patch.object(search, "_wiki_find_title", side_effect=ConnectionError("offline")):
            output = search.wikipedia_text("tesla")
        self.assertIn("Wikipedia search failed", output)

    def test_summary_helper_builds_fallback_url(self):
        response = mock.Mock()
        response.json.return_value = {"title": "Marie Curie", "extract": "Physicist."}
        with mock.patch.object(search.requests, "get", return_value=response):
            page = search._wiki_summary("Marie Curie")
        self.assertEqual(page["url"], "https://en.wikipedia.org/wiki/Marie_Curie")

    def test_find_title_reads_first_hit(self):
        response = mock.Mock()
        response.json.return_value = {"query": {"search": [{"title": "Python (programming language)"}]}}
        with mock.patch.object(search.requests, "get", return_value=response) as get:
            title = search._wiki_find_title("python language")
        self.assertEqual(title, "Python (programming language)")
        self.assertEqual(get.call_args.kwargs["params"]["srsearch"], "python language")


if __name__ == "__main__":
    unittest.main()
