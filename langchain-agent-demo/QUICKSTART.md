# 🚀 Quick Start Guide

Get up and running with the LangChain Agent Demo in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Installation

### 1. Navigate to the project directory

```bash
cd langchain-agent-demo
```

### 2. Run the setup script

```bash
python setup.py
```

This will:
- Check your Python version
- Create a `.env` file
- Offer to install dependencies
- Set up necessary directories

### 3. Configure API Keys

Edit the `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 4. (Optional) Install dependencies manually

If you skipped installation during setup:

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Demo

Launch the interactive menu to explore all examples:

```bash
python main.py
```

This provides a user-friendly interface to navigate through all examples.

### Run Specific Examples

Run individual modules directly:

```bash
# Basic chains
python -m src.01_basics.chains

# Agent with tools
python -m src.02_agents.simple_agent

# LangGraph
python -m src.04_langgraph.simple_graph

# Multi-agent system
python -m src.05_multi_agent.research_team

# RAG system
python -m src.06_rag.qa_system
```

### Run All Examples

Execute all examples sequentially:

```bash
python run_all_examples.py
```

## Learning Path

### Beginner Path (Start Here!)

1. **Basic Chains** (`src.01_basics.chains`)
   - Learn how chains work
   - Understand LCEL syntax
   - Simple prompt → LLM → output

2. **Prompt Engineering** (`src.01_basics.prompts`)
   - Master prompt templates
   - Few-shot learning
   - System messages

3. **Simple Agents** (`src.02_agents.simple_agent`)
   - First agent with tools
   - Tool selection
   - Basic reasoning

4. **LangGraph Basics** (`src.04_langgraph.simple_graph`)
   - State machines
   - Nodes and edges
   - Simple workflows

### Intermediate Path

5. **LLM Interactions** (`src.01_basics.llm_examples`)
   - Streaming
   - Token tracking
   - Async operations

6. **Conversation Memory** (`src.03_memory.conversation_memory`)
   - Add context to conversations
   - Different memory types
   - Stateful interactions

7. **Conditional Graphs** (`src.04_langgraph.conditional_graph`)
   - Dynamic routing
   - Conditional edges
   - Cyclic workflows

8. **RAG System** (`src.06_rag.qa_system`)
   - Document loading
   - Vector search
   - Question answering

### Advanced Path

9. **ReAct Agents** (`src.02_agents.react_agent`)
   - Advanced reasoning
   - Tool chaining
   - Error handling

10. **Multi-Agent Systems** (`src.05_multi_agent.research_team`)
    - Agent coordination
    - Supervisor patterns
    - Collaborative workflows

## Common Issues

### API Key Not Set

**Error:** "OPENAI_API_KEY not set"

**Solution:** 
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key
3. Make sure `.env` is in the project root

### Import Errors

**Error:** "ModuleNotFoundError: No module named 'langchain'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Python Version

**Error:** "Python 3.10 or higher is required"

**Solution:**
- Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
- Use a virtual environment: `python3.10 -m venv venv`

## Tips

1. **Start with the interactive demo** (`python main.py`) - it's the easiest way to explore

2. **Read the code comments** - each example is extensively documented

3. **Experiment** - modify parameters and see what happens

4. **Check the README** - for detailed documentation of all features

5. **Use verbose mode** - many examples have verbose output showing the agent's thought process

## Next Steps

- Explore the `examples/` directory for additional resources
- Read module docstrings for detailed explanations
- Check out the [LangChain documentation](https://python.langchain.com/)
- Try building your own agents!

## Need Help?

- Check `README.md` for comprehensive documentation
- Read inline code comments in each module
- Each example includes educational explanations
- Look at the docstrings in the source code

---

**Happy Learning! 🎓**

Now run `python main.py` to get started!

