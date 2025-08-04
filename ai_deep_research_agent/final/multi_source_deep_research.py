import asyncio
import streamlit as st
from typing import Dict, Any, List
from agents import Agent, Runner, trace
from agents import set_default_openai_key
from firecrawl import FirecrawlApp
from newsapi import NewsApiClient
import arxiv
import praw
from serpapi import GoogleSearch
from tenacity import retry, stop_after_attempt, wait_exponential
from agents.tool import function_tool

# Set page configuration
st.set_page_config(
    page_title="OpenAI Multi-Source Deep Research Agent",
    page_icon="📘",
    layout="wide"
)

# Initialize session state for API keys if not exists
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = ""
if "newsapi_api_key" not in st.session_state:
    st.session_state.newsapi_api_key = ""
if "serper_api_key" not in st.session_state:
    st.session_state.serper_api_key = ""
if "reddit_client_id" not in st.session_state:
    st.session_state.reddit_client_id = ""
if "reddit_client_secret" not in st.session_state:
    st.session_state.reddit_client_secret = ""
if "reddit_user_agent" not in st.session_state:
    st.session_state.reddit_user_agent = ""


# Sidebar for API keys
with st.sidebar:
    st.title("API Configuration")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password"
    )
    firecrawl_api_key = st.text_input(
        "Firecrawl API Key",
        value=st.session_state.firecrawl_api_key,
        type="password"
    )
    newsapi_api_key = st.text_input(
        "NewsAPI Key",
        value=st.session_state.newsapi_api_key,
        type="password"
    )
    serper_api_key = st.text_input(
        "Serper API Key",
        value=st.session_state.serper_api_key,
        type="password"
    )
    st.subheader("Reddit API")
    reddit_client_id = st.text_input(
        "Client ID",
        value=st.session_state.reddit_client_id,
        type="password"
    )
    reddit_client_secret = st.text_input(
        "Client Secret",
        value=st.session_state.reddit_client_secret,
        type="password"
    )
    reddit_user_agent = st.text_input(
        "User Agent",
        value=st.session_state.reddit_user_agent,
    )

    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        set_default_openai_key(openai_api_key)
    if firecrawl_api_key:
        st.session_state.firecrawl_api_key = firecrawl_api_key
    if newsapi_api_key:
        st.session_state.newsapi_api_key = newsapi_api_key
    if serper_api_key:
        st.session_state.serper_api_key = serper_api_key
    if reddit_client_id:
        st.session_state.reddit_client_id = reddit_client_id
    if reddit_client_secret:
        st.session_state.reddit_client_secret = reddit_client_secret
    if reddit_user_agent:
        st.session_state.reddit_user_agent = reddit_user_agent

# Main content
st.title("📘 OpenAI Multi-Source Deep Research Agent")
st.markdown("This OpenAI Agent performs deep research on any topic using multiple sources.")

# Research topic input
research_topic = st.text_input("Enter your research topic:", placeholder="e.g., Latest developments in AI")

# Data source selection
sources = st.multiselect(
    "Select data sources:",
    options=["firecrawl", "newsapi", "arxiv", "google", "reddit"],
    default=["firecrawl"]
)

# --- Data Source Tools ---

@function_tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def firecrawl_deep_research(query: str, max_depth: int = 2, time_limit: int = 120, max_urls: int = 5) -> Dict[str, Any]:
    """
    Perform comprehensive web research using Firecrawl's deep research endpoint.
    This tool is best for deep, exploratory research on a topic.
    """
    try:
        firecrawl_app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
        params = {"maxDepth": max_depth, "timeLimit": time_limit, "maxUrls": max_urls}
        
        def on_activity(activity):
            st.write(f"[{activity['type']}] {activity['message']}")
            
        with st.spinner("Performing Firecrawl deep research..."):
            results = firecrawl_app.deep_research(query=query, params=params, on_activity=on_activity)
        
        if 'data' in results and 'finalAnalysis' in results['data']:
            return {
                "success": True,
                "summary": results['data']['finalAnalysis'],
                "sources": results['data']['sources']
            }
        else:
            return {"success": False, "error": "Invalid response from Firecrawl"}
    except Exception as e:
        return {"error": f"Firecrawl error: {str(e)}", "success": False}

@function_tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def newsapi_search(query: str, num_articles: int = 5) -> Dict[str, Any]:
    """
    Search for recent news articles on a given topic using NewsAPI.
    This tool is best for getting up-to-date news and current events.
    """
    try:
        newsapi = NewsApiClient(api_key=st.session_state.newsapi_api_key)
        with st.spinner("Searching for news articles..."):
            top_headlines = newsapi.get_top_headlines(q=query, language='en', page_size=num_articles)
        
        articles = []
        if top_headlines.get('status') == 'ok':
            for article in top_headlines.get('articles', []):
                articles.append({
                    "title": article.get('title', 'N/A'),
                    "url": article.get('url', '#'),
                    "summary": article.get('description', 'No summary available.')
                })
            return {"success": True, "articles": articles}
        else:
            return {"success": False, "error": top_headlines.get('message', 'Unknown error from NewsAPI')}
    except Exception as e:
        return {"error": f"NewsAPI error: {str(e)}", "success": False}

@function_tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def arxiv_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for academic papers on arXiv.
    This tool is best for finding scientific papers, pre-prints, and academic research.
    """
    try:
        with st.spinner("Searching arXiv for academic papers..."):
            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            results = list(search.results())
        
        papers = []
        for result in results:
            papers.append({
                "title": result.title,
                "url": result.pdf_url,
                "summary": result.summary
            })
            
        return {"success": True, "papers": papers}
    except Exception as e:
        return {"error": f"ArXiv error: {str(e)}", "success": False}

@function_tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def google_search(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Perform a Google search using the Serper API.
    This tool is best for general web searches and finding a variety of sources.
    """
    try:
        with st.spinner("Performing Google search..."):
            search = GoogleSearch({
                "q": query,
                "num": num_results,
                "api_key": st.session_state.serper_api_key
            })
            results = search.get_dict()
        
        search_results = []
        if 'organic_results' in results:
            for result in results['organic_results']:
                search_results.append({
                    "title": result.get('title', 'N/A'),
                    "url": result.get('link', '#'),
                    "summary": result.get('snippet', 'No summary available.')
                })
        return {"success": True, "results": search_results}
    except Exception as e:
        return {"error": f"Google Search (Serper) error: {str(e)}", "success": False}

@function_tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def reddit_search(query: str, num_posts: int = 5) -> Dict[str, Any]:
    """
    Search for discussions on Reddit.
    This tool is best for finding opinions, discussions, and community feedback.
    """
    try:
        with st.spinner("Searching Reddit..."):
            reddit = praw.Reddit(
                client_id=st.session_state.reddit_client_id,
                client_secret=st.session_state.reddit_client_secret,
                user_agent=st.session_state.reddit_user_agent
            )
            subreddit = reddit.subreddit("all")
            posts = list(subreddit.search(query, limit=num_posts))

        search_results = []
        for post in posts:
            search_results.append({
                "title": post.title,
                "url": post.url,
                "summary": post.selftext[:200] + "..." if post.selftext else "No text content."
            })
        return {"success": True, "posts": search_results}
    except Exception as e:
        return {"error": f"Reddit error: {str(e)}", "success": False}

# --- Agents ---

strategy_agent = Agent(
    name="strategy_agent",
    instructions="""You are a research strategy expert. Your job is to analyze a research
    topic and create a plan for how to best research it using the available tools.

    When given a research topic, you must:
    1.  Determine the primary category of the research topic. Choose from:
        - "ACADEMIC_SCIENTIFIC"
        - "NEWS_CURRENT_EVENTS"
        - "TECHNICAL_PROGRAMMING"
        - "GENERAL_OVERVIEW"
    2.  Based on the category, select the most appropriate data sources to use.
        - For "ACADEMIC_SCIENTIFIC": Prioritize arxiv, google.
        - For "NEWS_CURRENT_EVENTS": Prioritize newsapi, google, reddit.
        - For "TECHNICAL_PROGRAMMING": Prioritize google, firecrawl, reddit.
        - For "GENERAL_OVERVIEW": Prioritize firecrawl, google, newsapi.
    3.  Generate a JSON object containing the `category`, a `strategy_summary`, and a `research_plan`.
        The `research_plan` should be a list of dictionaries, where each dictionary represents
        a tool call with the `source_name` and optimized `parameters` (like a more specific `query`
        or filters).

    Example Output:
    {
      "category": "ACADEMIC_SCIENTIFIC",
      "strategy_summary": "This is an academic topic, so I will focus on arXiv for papers and Google for broader academic search.",
      "research_plan": [
        {
          "source_name": "arxiv",
          "parameters": {"query": "latest breakthroughs in quantum machine learning", "max_results": 5}
        },
        {
          "source_name": "google",
          "parameters": {"query": "quantum machine learning review article", "num_results": 7}
        }
      ]
    }
    """
)

research_agent = Agent(
    name="research_agent",
    instructions="""You are a research orchestrator. Your job is to execute a given research plan.
    You will be given a list of tool calls to make. You must execute each tool call as specified
    in the plan and return the raw, structured output from each tool.
    """,
    tools=[] # Tools will be added dynamically
)

synthesis_agent = Agent(
    name="synthesis_agent",
    instructions="""You are a synthesis expert. Your job is to create a coherent report 
    from a ranked list of information from various sources.

    When given a ranked list of research results:
    1.  Analyze the provided information, paying attention to the source and score of each item.
    2.  Construct a well-structured research report that synthesizes the key findings.
    3.  **Crucially, if you find conflicting or contradictory information between sources, you must
        highlight this conflict.** State which sources are in disagreement and what the conflicting
        points are. For example: "Source A (e.g., NewsAPI) reports X, while Source B (e.g., Reddit)
        suggests Y, which contradicts the finding from Source A."
    4.  Cite your sources for each key point using the format [Source: Title].
    5.  The final output should be a comprehensive and clear markdown report.
    """
)

elaboration_agent = Agent(
    name="elaboration_agent",
    instructions="""You are an expert content enhancer specializing in research elaboration.
    
    When given a research report:
    1. Analyze the structure and content of the report.
    2. Enhance the report by:
       - Adding more detailed explanations of complex concepts.
       - Including relevant examples, case studies, and real-world applications.
       - Expanding on key points with additional context and nuance.
       - Adding visual elements descriptions (charts, diagrams, infographics).
       - Incorporating latest trends and future predictions.
       - Suggesting practical implications for different stakeholders.
    3. Maintain academic rigor and factual accuracy.
    4. Preserve the original structure while making it more comprehensive.
    5. Ensure all additions are relevant and valuable to the topic.
    """
)
# In the sidebar
with st.sidebar:
    st.title("Research Strategy")
    strategy_mode = st.radio(
        "Source Selection",
        ("Automatic", "Manual"),
        index=0, # Default to Automatic
        help="**Automatic**: The agent decides the best sources based on your topic. \n**Manual**: You choose the sources to use."
    )

# In the main UI body
st.markdown("### 2. Select Data Sources")
if strategy_mode == "Manual":
    sources = st.multiselect(
        "Select data sources:",
        options=["firecrawl", "newsapi", "arxiv", "google", "reddit"],
        default=["firecrawl"]
    )
else:
    st.info("🤖 **Automatic Mode**: The agent will choose the best sources for your topic.")
    sources = [] # Will be determined by the strategy agent


async def run_research_process(topic: str, selected_sources: List[str]):
    """Run the complete research process."""
    
    # Dynamically assign tools to the research agent
    research_agent.tools = [available_tools[source] for source in selected_sources]
    
    # Step 1: Initial Research (Data Gathering)
    with st.spinner("Conducting initial research... This may take a moment."):
        # The prompt for the research agent is simple: just execute the tools.
        research_prompt = f"Gather information on '{topic}' using the available tools."
        
        # We expect the agent to return a dictionary of results from the tools it called.
        # Note: The 'agents' library might wrap the final output in a specific object.
        research_results_obj = await Runner.run(research_agent, research_prompt)
        
        # Extract the actual dictionary from the agent's output.
        # This might need adjustment depending on the exact structure of the Agent/Runner output.
        raw_results = research_results_obj.final_output if hasattr(research_results_obj, 'final_output') else {}


    # Step 2: Aggregate and Rank
    with st.spinner("Aggregating and ranking results..."):
        ranked_results = aggregate_and_rank_results(raw_results, topic)
        # For transparency, show the ranked results in the UI
        with st.expander("View Ranked and Deduplicated Results"):
            st.json(ranked_results)

    # Step 3: Synthesize Initial Report
    with st.spinner("Synthesizing the initial report..."):
        synthesis_prompt = f"""
        Here is a ranked and deduplicated list of research results for the topic '{topic}'.
        Please synthesize this information into a coherent report. Make sure to highlight
        any conflicting information you find between sources.

        Ranked Results:
        {ranked_results}
        """
        synthesis_result = await Runner.run(synthesis_agent, synthesis_prompt)
        initial_report = synthesis_result.final_output

    # Display initial report in an expander
    with st.expander("View Initial Research Report"):
        st.markdown(initial_report)
    
    # Step 4: Enhance the report
    with st.spinner("Enhancing the report with additional information..."):
        elaboration_input = f"""
        RESEARCH TOPIC: {topic}
        
        INITIAL RESEARCH REPORT:
        {initial_report}
        
        Please enhance this research report with additional information, examples, case studies, 
        and deeper insights while maintaining its academic rigor and factual accuracy.
        """
        elaboration_result = await Runner.run(elaboration_agent, elaboration_input)
        enhanced_report = elaboration_result.final_output
    
    return enhanced_report

# Main research process
if st.button("Start Research", disabled=not (openai_api_key and research_topic and sources)):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not research_topic:
        st.warning("Please enter a research topic.")
    elif not sources:
        st.warning("Please select at least one data source.")
    else:
        # Check for necessary API keys
        required_keys = {
            "firecrawl": st.session_state.firecrawl_api_key,
            "newsapi": st.session_state.newsapi_api_key,
            "google": st.session_state.serper_api_key,
            "reddit": all([st.session_state.reddit_client_id, st.session_state.reddit_client_secret, st.session_state.reddit_user_agent])
        }
        
        missing_keys = [source for source in sources if source in required_keys and not required_keys[source]]
        
        if missing_keys:
            st.warning(f"Please enter the API key(s) for the following sources in the sidebar: {', '.join(missing_keys)}")
        else:
            try:
                # Create placeholder for the final report
                report_placeholder = st.empty()
                
                # Run the research process
                enhanced_report = asyncio.run(run_research_process(research_topic, sources))
                
                # Display the enhanced report
                report_placeholder.markdown("## Enhanced Research Report")
                report_placeholder.markdown(enhanced_report)
                
                # Add download button
                st.download_button(
                    "Download Report",
                    enhanced_report,
                    file_name=f"{research_topic.replace(' ', '_')}_report.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# Footer
st.markdown("---")
st.markdown("Powered by OpenAI Agents SDK and Firecrawl") 