import os
import asyncio
import threading
import requests
import aiosqlite
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'Chatbot with tools & MCP'

# ---------- Background Async Loop Setup ----------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

# ---------- LLM ----------
llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.6)

# ---------- Local Tools ----------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

# ---------- MCP Client & Tools ----------
client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",  # Fallback to "sse" if server requires SSE
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print(f"Warning: Failed to load MCP tools: {e}")
        return []

mcp_tools = load_mcp_tools()

# Fixed: Added calculator tool into tools list
tools = [search_tool, calculator, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# ---------- State Definition ----------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------- Graph Nodes ----------
async def chat_node(state: ChatState):
    """LLM node that may answer directly or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

# ---------- Async Checkpointer ----------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# ---------- Graph Assembly ----------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# ---------- Helper Functions ----------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if thread_id:
            all_threads.add(thread_id)
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())
   

