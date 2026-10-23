"""Checks the LangChain tool wrappers. Skipped when langchain-core is missing."""

import importlib.util
import unittest

HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None


@unittest.skipUnless(HAS_LANGCHAIN, "langchain-core is not installed")
class ToolWrapperTests(unittest.TestCase):
    def test_all_tools_are_registered(self):
        from src.agent.tools import ALL_TOOLS

        self.assertEqual(
            {t.name for t in ALL_TOOLS},
            {"get_current_time", "calculate", "web_search", "wikipedia_search", "search_documents"},
        )

    def test_every_tool_has_a_description(self):
        from src.agent.tools import ALL_TOOLS

        for tool in ALL_TOOLS:
            self.assertTrue(tool.description.strip(), tool.name)

    def test_calculate_tool(self):
        from src.agent.tools import calculate

        self.assertEqual(calculate.invoke({"a": 1234, "b": 5678, "operation": "multiply"}), "7006652")

    def test_calculate_tool_reports_errors_as_text(self):
        from src.agent.tools import calculate

        output = calculate.invoke({"a": 1, "b": 0, "operation": "divide"})
        self.assertTrue(output.startswith("Error:"))

    def test_time_tool_returns_text(self):
        from src.agent.tools import get_current_time

        self.assertTrue(get_current_time.invoke({}))


if __name__ == "__main__":
    unittest.main()
