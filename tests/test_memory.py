import tempfile
import unittest
from pathlib import Path

from src.agent.memory import DEFAULT_TITLE, ChatStore, make_title


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ChatStore(Path(self._tmp.name) / "nested" / "chat.db")

    def test_create_and_list_conversations(self):
        first = self.store.create_conversation()
        second = self.store.create_conversation("Second")
        ids = [c["id"] for c in self.store.list_conversations()]
        self.assertEqual(set(ids), {first, second})
        self.assertEqual(self.store.get_conversation(second)["title"], "Second")
        self.assertIsNone(self.store.get_conversation(999))

    def test_first_user_message_sets_title(self):
        conv = self.store.create_conversation()
        self.store.add_message(conv, "user", "What is the capital of France?")
        self.store.add_message(conv, "assistant", "Paris.")
        self.store.add_message(conv, "user", "And Italy?")
        self.assertEqual(self.store.get_conversation(conv)["title"], "What is the capital of France?")

    def test_custom_title_is_kept(self):
        conv = self.store.create_conversation("My research")
        self.store.add_message(conv, "user", "hello")
        self.assertEqual(self.store.get_conversation(conv)["title"], "My research")

    def test_messages_round_trip_with_tools(self):
        conv = self.store.create_conversation()
        self.store.add_message(conv, "user", "Latest news?")
        self.store.add_message(conv, "assistant", "Here it is.", ["web_search"])
        messages = self.store.get_messages(conv)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["tools_used"], [])
        self.assertEqual(messages[1]["tools_used"], ["web_search"])

    def test_history_is_limited_and_plain(self):
        conv = self.store.create_conversation()
        for i in range(6):
            self.store.add_message(conv, "user", f"q{i}")
            self.store.add_message(conv, "assistant", f"a{i}")
        history = self.store.history(conv, limit=4)
        self.assertEqual(history, [
            {"role": "user", "content": "q4"},
            {"role": "assistant", "content": "a4"},
            {"role": "user", "content": "q5"},
            {"role": "assistant", "content": "a5"},
        ])

    def test_history_starts_with_a_user_message(self):
        conv = self.store.create_conversation()
        self.store.add_message(conv, "user", "q1")
        self.store.add_message(conv, "assistant", "a1")
        self.store.add_message(conv, "user", "q2")
        history = self.store.history(conv, limit=2)  # window would start on the assistant
        self.assertEqual(history, [{"role": "user", "content": "q2"}])

    def test_delete_conversation_removes_messages(self):
        conv = self.store.create_conversation()
        self.store.add_message(conv, "user", "hi")
        self.store.delete_conversation(conv)
        self.assertIsNone(self.store.get_conversation(conv))
        self.assertEqual(self.store.get_messages(conv), [])

    def test_data_survives_a_new_store_instance(self):
        conv = self.store.create_conversation()
        self.store.add_message(conv, "user", "remember me")
        reopened = ChatStore(self.store.db_path)
        self.assertEqual(reopened.get_messages(conv)[0]["content"], "remember me")

    def test_rename(self):
        conv = self.store.create_conversation()
        self.store.rename_conversation(conv, "  Renamed   chat ")
        self.assertEqual(self.store.get_conversation(conv)["title"], "Renamed chat")


class MakeTitleTests(unittest.TestCase):
    def test_short_text_kept(self):
        self.assertEqual(make_title("Hello   world"), "Hello world")

    def test_long_text_shortened(self):
        title = make_title("x" * 200)
        self.assertEqual(len(title), 50)
        self.assertTrue(title.endswith("…"))

    def test_empty_text_uses_default(self):
        self.assertEqual(make_title("   "), DEFAULT_TITLE)


if __name__ == "__main__":
    unittest.main()
