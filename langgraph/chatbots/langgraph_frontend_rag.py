import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph_backend_rag import (chatbot, ingest_pdf, retrieve_all_threads, thread_document_metadata)

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Render chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_container = st.empty()
        text_placeholder = st.empty()
        full_response = ""
        status_box = None

        for message_chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            # Handle tool status display
            if isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, "name", "tool")
                if status_box is None:
                    status_box = status_container.status(
                        f"🔧 Using `{tool_name}` …", expanded=True
                    )
                else:
                    status_box.update(
                        label=f"🔧 Using `{tool_name}` …",
                        state="running",
                        expanded=True,
                    )

            # Handle text streaming
            elif isinstance(message_chunk, AIMessage):
                content = message_chunk.content
                text_chunk = ""
                
                if isinstance(content, str):
                    text_chunk = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            text_chunk += block["text"]
                        elif isinstance(block, str):
                            text_chunk += block

                if text_chunk:
                    full_response += text_chunk
                    text_placeholder.markdown(full_response + "▌")

        # Finalize status box & lock in text output
        if status_box is not None:
            status_box.update(
                label="✅ Tool finished", state="complete", expanded=False
            )
        
        text_placeholder.markdown(full_response)

    # Save complete response to session history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )