import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_backend_mcp import chatbot, retrieve_all_threads, run_async

# **************************************** Utility Functions *************************

def generate_thread_id() -> str:
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id: str):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id: str):
    # Async checkpointer requires aget_state executed on the backend loop
    state = run_async(chatbot.aget_state(config={'configurable': {'thread_id': thread_id}}))
    messages = state.values.get('messages', [])
    
    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({'role': 'user', 'content': msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            # Filter out intermediate tool-call metadata
            temp_messages.append({'role': 'assistant', 'content': msg.content})
        elif isinstance(msg, ToolMessage):
            # Skip raw tool outputs (JSON/dicts)
            continue
            
    return temp_messages


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph + MCP Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id), key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = load_conversation(thread_id)


# **************************************** Main UI ************************************

# Render chat history with Markdown support
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here...')

if user_input:
    # 1. Display and save user input
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    # 2. Invoke model directly via run_async (No streaming queue bridge)
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            response = run_async(
                chatbot.ainvoke(
                    {'messages': [HumanMessage(content=user_input)]},
                    config=CONFIG
                )
            )
            ai_message = response["messages"][-1].content
            st.markdown(ai_message)

    # 3. Store final response in session history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})