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

#another way of implementing browser: Browser | None
browser = Union[Browser, None] 
page = Union[Page, None]


#defining the state dictionary which will be passed between nodes
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] #for conversation history
    url: Union[str, None]
    current_ss: Union[List[str], None]
    summaries: Annotated[Sequence[BaseMessage], add_messages]
    scroll_decision: Union[str, None]
    task: str


async def initialize_browser():
    """
    Initialize the playwright browser and page
    """

    global browser, page
    print('-----Initializing the Playwright browser-----')

    try:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless = False)

        page = await browser.new_page()
        print('-----Browser Initialized-----')

    except Exception as e:
        print(f'Failed to initialized browser due the following exception: {e}')
