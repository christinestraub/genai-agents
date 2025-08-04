import asyncio
import streamlit as st
from typing import Dict, Any, List
from agents import Agent, Runner
from firecrawl import FirecrawlApp
from newsapi import NewsApiClient
import arxiv
import praw
from serpapi import GoogleSearch
from tenacity import retry, stop_after_attempt, wait_exponential
from agents.tool import function_tool
import time
import json

# --- Caching Configuration ---
# Use Streamlit's caching mechanism. Each cache has a Time-To-Live (TTL).
@st.cache_data(ttl=3600) # Cache for 1 hour
def firecrawl_deep_research_cached(**kwargs):
    return FirecrawlApp(api_key=st.session_state.firecrawl_api_key).deep_research(**kwargs)

@st.cache_data(ttl=600) # Cache for 10 minutes
def newsapi_search_cached(**kwargs):
    return NewsApiClient(api_key=st.session_state.newsapi_api_key).get_top_headlines(**kwargs)

@st.cache_data(ttl=86400) # Cache for 24 hours
def arxiv_search_cached(**kwargs):
    return list(arxiv.Search(**kwargs).results())

@st.cache_data(ttl=3600) # Cache for 1 hour
def google_search_cached(**kwargs):
    return GoogleSearch(kwargs).get_dict()

@st.cache_data(ttl=1800) # Cache for 30 minutes
def reddit_search_cached(query, limit):
    reddit = praw.Reddit(
        client_id=st.session_state.reddit_client_id,
        client_secret=st.session_state.reddit_client_secret,
        user_agent=st.session_state.reddit_user_agent
    )
    subreddit = reddit.subreddit("all")
    return list(subreddit.search(query, limit=limit))


# --- Optimized Data Source Tools (with Caching) ---
@function_tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def firecrawl_deep_research(query: str, max_depth: int = 2, time_limit: int = 120, max_urls: int = 5) -> Dict[str, Any]:
    # ... (implementation remains the same, but calls the cached function)
    pass # Placeholder

@function_tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def newsapi_search(query: str, num_articles: int = 5) -> Dict[str, Any]:
    # ... (implementation remains the same, but calls the cached function)
    pass # Placeholder

# (Other tools will also be updated to use caching)
# ...

# --- Aggregation and Synthesis (Optimized) ---
def aggregate_and_rank_results(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    # ... (Deduplication can be optimized if it becomes a bottleneck)
    pass # Placeholder


# --- Agents ---
strategy_agent = Agent(...)
research_agent = Agent(...)
synthesis_agent = Agent(...)
elaboration_agent = Agent(...)

# --- Main Process Logic (Parallel and Optimized) ---
async def run_research_process(topic: str, research_plan: List[Dict[str, Any]]):
    """
    Runs the research for each source in parallel and aggregates the results.
    """
    tasks = []
    for step in research_plan:
        source_name = step["source_name"]
        parameters = step["parameters"]
        tool = available_tools[source_name]
        tasks.append(asyncio.create_task(tool(**parameters)))
    
    # Gather results from all sources concurrently
    source_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    # ...
    return # final report

available_tools = {
    "firecrawl": firecrawl_deep_research,
    "newsapi": newsapi_search,
    "arxiv": arxiv_search,
    "google": google_search,
    "reddit": reddit_search,
}
