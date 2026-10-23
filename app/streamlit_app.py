"""Streamlit web interface for the AI Research Assistant.

Run from the project root:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.agent import config  # noqa: E402  (also loads the .env file)
from src.agent.agent import MissingApiKeyError, build_agent, run_reply, stream_events  # noqa: E402
from src.agent.memory import ChatStore  # noqa: E402
from src.agent.rag import SUPPORTED_EXTENSIONS, get_store, save_upload  # noqa: E402
from src.agent.utils import setup_logging  # noqa: E402

st.set_page_config(page_title=config.APP_NAME, page_icon="🔎")
setup_logging()


@st.cache_resource
def load_agent():
    return build_agent()


@st.cache_resource
def load_chat_store():
    return ChatStore()


try:
    agent = load_agent()
except MissingApiKeyError as exc:
    st.error(str(exc))
    st.stop()

store = load_chat_store()
documents = get_store()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


# ---------- sidebar ----------

with st.sidebar:
    st.title(config.APP_NAME)
    if st.button("➕ New chat"):
        st.session_state.conversation_id = None
        st.rerun()

    st.subheader("Chats")
    conversations = store.list_conversations(15)
    if not conversations:
        st.caption("No saved chats yet.")
    for item in conversations:
        if st.button(item["title"], key=f"chat-{item['id']}"):
            st.session_state.conversation_id = item["id"]
            st.rerun()

    st.subheader("Documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT or Markdown",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    if uploads and st.button("Index uploaded files"):
        for upload in uploads:
            try:
                with st.spinner(f"Indexing {upload.name}..."):
                    path = save_upload(upload.name, upload.getvalue())
                    count = documents.add_document(path)
                st.success(f"{upload.name}: {count} chunks")
            except Exception as exc:
                st.error(f"{upload.name}: {exc}")

    try:
        indexed = documents.list_documents()
    except Exception as exc:
        indexed = {}
        st.caption(f"Document index unavailable: {exc}")
    for name, chunks in indexed.items():
        left, right = st.columns([4, 1])
        left.caption(f"{name} ({chunks} chunks)")
        if right.button("✖", key=f"remove-{name}", help=f"Remove {name}"):
            documents.delete_document(name)
            st.rerun()


# ---------- chat area ----------

st.header("Research Assistant")
st.caption("Ask anything. I can search the web and Wikipedia, read your uploaded documents, and do maths.")

conversation_id = st.session_state.conversation_id

if conversation_id is not None:
    for message in store.get_messages(conversation_id):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["tools_used"]:
                st.caption("Tools used: " + ", ".join(message["tools_used"]))
    if st.button("🗑 Delete this chat"):
        store.delete_conversation(conversation_id)
        st.session_state.conversation_id = None
        st.rerun()

prompt = st.chat_input("Ask a question...")

if prompt:
    history = store.history(conversation_id) if conversation_id is not None else []
    messages = history + [{"role": "user", "content": prompt}]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        tools_used: list[str] = []
        try:
            if config.STREAMING:

                def token_stream():
                    for event in stream_events(agent, messages):
                        if event.kind == "tool":
                            if event.value not in tools_used:
                                tools_used.append(event.value)
                            status.caption(f"🔧 Using {event.value}...")
                        else:
                            yield event.value

                answer = st.write_stream(token_stream())
            else:
                with st.spinner("Thinking..."):
                    reply = run_reply(agent, messages)
                answer, tools_used = reply.text, reply.tools_used
                st.markdown(answer)
        except Exception as exc:
            status.empty()
            st.error(f"Something went wrong: {exc}")
            st.stop()

        status.caption("Tools used: " + ", ".join(tools_used) if tools_used else "")

    if conversation_id is None:
        conversation_id = store.create_conversation()
        st.session_state.conversation_id = conversation_id
    store.add_message(conversation_id, "user", prompt)
    store.add_message(conversation_id, "assistant", answer, tools_used)
    st.rerun()
