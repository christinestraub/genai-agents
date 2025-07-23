# Visual Automation Summarization Agent

This project implements a **Visual Web Browser Agent** using LangGraph and Playwright. The agent can visually browse a web page, take screenshots, summarize content using a vision-capable LLM, and decide whether to scroll for more information, all in an automated workflow.

## Features
- **Automated Browser Control:** Uses Playwright to launch and control a Chromium browser.
- **Visual Summarization:** Takes screenshots of the current page and summarizes them using OpenAI's GPT-4o LLM via LangChain.
- **Iterative Browsing:** Decides whether to scroll for more content and repeats the screenshot/summarization loop as needed.
- **Stateful Workflow:** Built with LangGraph, modeling the agent as a directed graph of nodes (steps) and edges (control flow).
- **Extensible:** Easily add new tools or nodes for more complex browser automation tasks.

## How It Works
1. **Initialize Browser:** Launches a Playwright browser and navigates to a target URL (default: IBM's page about Large Language Models).
2. **Screenshot & Summarize:** Takes a screenshot, sends it to a vision LLM, and stores the summary.
3. **Scroll Decision:** The agent decides whether to scroll for more content (currently forced to always scroll for demo/testing).
4. **Loop or Aggregate:** If scrolling, repeats the screenshot/summarization. If not, aggregates all summaries into a final report.
5. **Clean Up:** Closes the browser after execution.

## Project Structure
```
visual-automation-summarization/
  └── agentic_brwoser.py  # Main agent code
```

## Requirements
- Python 3.8+
- [Playwright](https://playwright.dev/python/)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [langchain-openai](https://python.langchain.com/docs/integrations/llms/openai)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

## Installation
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd genai-agents/visual-automation-summarization
   ```
2. **Install dependencies:**
   ```sh
   pip install playwright langchain langgraph langchain-openai python-dotenv
   playwright install
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the project directory with your OpenAI API key:
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage
Run the agent with:
```sh
python agentic_brwoser.py
```

The agent will:
- Open a browser window
- Navigate to IBM's comprehensive page about Large Language Models
- Take screenshots, summarize them, and decide whether to scroll
- Aggregate the summaries into a final report
- Print progress and results to the console

## Customization
- **Change the target URL or task:** Edit the `initial_state` dictionary in `agentic_brwoser.py`.
- **Modify scroll logic:** Update the `route_scroll_decision` function to use LLM decisions instead of always scrolling.
- **Add new tools or nodes:** Use the `@tool` decorator and add nodes/edges in the LangGraph workflow.

## References
- [LangGraph Documentation](https://www.langchain.com/resources)
- [Playwright Python Docs](https://playwright.dev/python/)
- [LangChain Docs](https://python.langchain.com/)