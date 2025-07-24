# GenAI Agents Collection

A collection of intelligent AI agents built with modern frameworks like LangGraph, LangChain, and various automation tools. Each agent is designed to solve specific tasks through autonomous decision-making and tool usage.

## Visual Automation Summarization Agent
**Location:** `visual-automation-summarization/`

An intelligent web browsing agent that can visually navigate websites, take screenshots, and provide AI-powered summaries of web content.

**Key Features:**
- Automated browser control with Playwright
- Visual content analysis using GPT-4o
- Intelligent scrolling decisions
- Comprehensive content summarization
- Stateful workflow management with LangGraph

**[📖 Read Full Documentation →](./visual-automation-summarization/README.md)**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Common Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd genai-agents
   ```

2. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Choose and run an agent:**
   Navigate to the specific agent directory and follow its individual README instructions.

## 📁 Project Structure

```
genai-agents/
├── README.md                           # This file - project overview
├── .env                               # Environment variables (create this)
├── visual-automation-summarization/   # Visual web browsing agent
│   ├── agentic_brwoser.py
│   └── README.md
└── [future-agent-directories]/        # Additional agents will be added here
```

## 🛠 Technology Stack

- **Agent Framework:** [LangGraph](https://github.com/langchain-ai/langgraph) - For building stateful, multi-actor applications
- **LLM Integration:** [LangChain](https://python.langchain.com/) + [OpenAI](https://openai.com/)
- **Browser Automation:** [Playwright](https://playwright.dev/python/)
- **Environment Management:** [python-dotenv](https://pypi.org/project/python-dotenv/)

## 🎯 Agent Design Philosophy

Each agent in this collection follows these principles:

- **Autonomous:** Agents can make decisions and execute actions independently
- **Stateful:** Built with LangGraph for complex workflow management
- **Tool-enabled:** Leverages specialized tools for specific tasks
- **Vision-capable:** Can process and understand visual content when needed
- **Extensible:** Modular design allows for easy customization and enhancement

## 🔮 Planned Agents

Future agents planned for this collection:

- **File Processing Agent:** Automated document analysis and processing
- **Data Analysis Agent:** Intelligent data exploration and insights generation
- **Code Review Agent:** Automated code analysis and suggestions
- **Research Agent:** Web research and information synthesis
- **Task Automation Agent:** General-purpose task automation

## 🤝 Contributing

Each agent is self-contained in its own directory with:
- Main implementation file(s)
- Dedicated README with specific instructions
- Any agent-specific dependencies or configuration

When adding a new agent:
1. Create a new directory with a descriptive name
2. Include a comprehensive README following the existing pattern
3. Update this main README to include your new agent
4. Ensure proper error handling and browser cleanup (if applicable)

## 📄 License

This project is open source. Please check individual agent directories for specific licensing information.

## 🔗 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Playwright Documentation](https://playwright.dev/python/)

---

**Note:** Each agent may have additional specific requirements. Please refer to individual agent READMEs for detailed setup and usage instructions. 
