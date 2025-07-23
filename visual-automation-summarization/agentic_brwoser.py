import base64, asyncio
from dotenv import load_dotenv
from typing import Annotated, Sequence, List, TypedDict, Union

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from playwright.async_api import async_playwright, Page, Browser


load_dotenv()