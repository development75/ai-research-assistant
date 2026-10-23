import unittest
from types import SimpleNamespace as NS

from src.agent.agent import Event, events_from_chunks, run_reply, tools_used_in_turn


def ai_chunk(text="", tool_calls=None):
    return NS(type="AIMessageChunk", content=text, tool_call_chunks=tool_calls or [])


def tool_message(name, call_id):
    return NS(type="tool", name=name, tool_call_id=call_id, content="result")


class EventsFromChunksTests(unittest.TestCase):
    def test_tokens_and_tools_in_order(self):
        stream = [
            (ai_chunk("", [{"name": "web_search", "id": "c1"}]), {}),
            (ai_chunk("", [{"name": None, "id": None}]), {}),        # later fragments carry no name
            (tool_message("web_search", "c1"), {}),                  # same call, must not repeat
            (ai_chunk("Hello "), {}),
            (ai_chunk("world"), {}),
        ]
        events = list(events_from_chunks(stream))
        self.assertEqual(events, [
            Event("tool", "web_search"),
            Event("token", "Hello "),
            Event("token", "world"),
        ])

    def test_tool_message_alone_is_reported(self):
        events = list(events_from_chunks([tool_message("calculate", "c9"), ai_chunk("42")]))
        self.assertEqual(events, [Event("tool", "calculate"), Event("token", "42")])

    def test_two_calls_to_the_same_tool_are_both_reported(self):
        stream = [tool_message("web_search", "c1"), tool_message("web_search", "c2")]
        self.assertEqual(len(list(events_from_chunks(stream))), 2)

    def test_human_messages_are_ignored(self):
        stream = [NS(type="human", content="my question"), ai_chunk("answer")]
        self.assertEqual(list(events_from_chunks(stream)), [Event("token", "answer")])

    def test_content_blocks_are_flattened(self):
        stream = [ai_chunk([{"type": "text", "text": "Hi"}])]
        self.assertEqual(list(events_from_chunks(stream)), [Event("token", "Hi")])

    def test_complete_message_used_when_nothing_streamed(self):
        stream = [NS(type="ai", content="Full answer", tool_call_chunks=[])]
        self.assertEqual(list(events_from_chunks(stream)), [Event("token", "Full answer")])

    def test_complete_message_not_duplicated_after_streaming(self):
        stream = [ai_chunk("Full "), ai_chunk("answer"), NS(type="ai", content="Full answer", tool_call_chunks=[])]
        tokens = [e.value for e in events_from_chunks(stream) if e.kind == "token"]
        self.assertEqual("".join(tokens), "Full answer")


class RunReplyTests(unittest.TestCase):
    def test_reply_text_and_tools(self):
        messages = [
            NS(type="human", content="q"),
            NS(type="ai", content="", name=None),
            NS(type="tool", name="calculate", content="4"),
            NS(type="tool", name="calculate", content="9"),
            NS(type="ai", content="The answer is 9."),
        ]

        class FakeAgent:
            def invoke(self, payload, config=None):
                self.payload, self.config = payload, config
                return {"messages": messages}

        agent = FakeAgent()
        reply = run_reply(agent, [{"role": "user", "content": "q"}])
        self.assertEqual(reply.text, "The answer is 9.")
        self.assertEqual(reply.tools_used, ["calculate"])
        self.assertIn("recursion_limit", agent.config)

    def test_tools_used_in_turn_deduplicates(self):
        messages = [NS(type="tool", name="a"), NS(type="tool", name="b"), NS(type="tool", name="a")]
        self.assertEqual(tools_used_in_turn(messages), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
