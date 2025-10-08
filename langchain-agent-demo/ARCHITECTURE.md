# 🏗️ Architecture Overview

This document explains the architecture and design principles of the LangChain Agent Demo.

## Project Structure

```
langchain-agent-demo/
├── config/                      # Configuration management
│   ├── __init__.py
│   └── settings.py             # Environment variables and settings
│
├── src/                        # Source code
│   ├── __init__.py
│   │
│   ├── 01_basics/              # Fundamental concepts
│   │   ├── chains.py           # Chain examples
│   │   ├── prompts.py          # Prompt engineering
│   │   └── llm_examples.py     # LLM interaction patterns
│   │
│   ├── 02_agents/              # Agent implementations
│   │   ├── custom_tools.py     # Custom tool creation
│   │   ├── simple_agent.py     # Basic agents
│   │   └── react_agent.py      # ReAct pattern agents
│   │
│   ├── 03_memory/              # Memory systems
│   │   └── conversation_memory.py  # Conversation context
│   │
│   ├── 04_langgraph/           # State machines
│   │   ├── simple_graph.py     # Basic graphs
│   │   └── conditional_graph.py # Conditional routing
│   │
│   ├── 05_multi_agent/         # Multi-agent systems
│   │   └── research_team.py    # Coordinated agents
│   │
│   ├── 06_rag/                 # RAG implementations
│   │   └── qa_system.py        # Question-answering
│   │
│   └── utils/                  # Utilities
│       ├── display.py          # Pretty printing
│       └── helpers.py          # Common functions
│
├── examples/                   # Sample data
│   └── sample_documents/
│
├── main.py                     # Interactive demo launcher
├── run_all_examples.py         # Batch executor
├── setup.py                    # Setup script
├── requirements.txt            # Dependencies
├── README.md                   # Main documentation
├── QUICKSTART.md              # Quick start guide
└── ARCHITECTURE.md            # This file
```

## Design Principles

### 1. Educational First

Every example is designed to teach:
- **Comprehensive Comments**: Code is extensively documented
- **Step-by-Step**: Examples break down complex concepts
- **Progressive Difficulty**: Start simple, build to advanced
- **Self-Contained**: Each example can run independently

### 2. Modular Architecture

```
┌─────────────────────────────────────────────────┐
│               User Interface Layer              │
│            (main.py, CLI menus)                 │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────┐
│            Example Modules Layer                │
│  (01_basics, 02_agents, 03_memory, etc.)       │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────┐
│           Core Utilities Layer                  │
│        (utils, config, helpers)                 │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────┐
│          LangChain/LangGraph Layer              │
│    (External framework dependencies)            │
└─────────────────────────────────────────────────┘
```

### 3. Separation of Concerns

- **Configuration**: Centralized in `config/settings.py`
- **Display Logic**: Isolated in `src/utils/display.py`
- **Business Logic**: In example modules
- **Tools**: Separate from agent logic in `custom_tools.py`

### 4. Reusability

Common patterns are abstracted:
- Prompt templates in utilities
- LLM loading helper functions
- Display formatting functions
- Tool factory functions

## Key Components

### Configuration System

```python
# config/settings.py
class Settings(BaseSettings):
    openai_api_key: str
    default_model: str
    temperature: float
    # ... and more
```

**Features:**
- Environment variable management
- Type validation with Pydantic
- Default values
- LangSmith integration

### Display Utilities

```python
# src/utils/display.py
- print_section()      # Section headers
- print_response()     # Format responses
- print_step()         # Step-by-step guides
- print_error()        # Error messages
```

**Benefits:**
- Consistent formatting
- Rich terminal output
- Easy to maintain
- Visually appealing

### Example Template

Each example follows this pattern:

```python
def example_N_descriptive_name():
    """
    Example N: Title
    Description of what this example demonstrates
    """
    print_section("Title", "Description")
    
    print_step(1, "First step")
    # Implementation
    
    print_step(2, "Second step")
    # Implementation
    
    print_response(result, "Output")

def main():
    """Run all examples."""
    example_1_...()
    example_2_...()
    # ...
```

## Data Flow Patterns

### 1. Simple Chain Flow

```
User Input → Prompt Template → LLM → Output Parser → Response
```

### 2. Agent Flow

```
User Query → Agent
              ↓
         [Reasoning Loop]
         ↓           ↓
      Thought    Action
         ↓           ↓
    Choose Tool  Execute
         ↓           ↓
    Observation  Result
         ↓           ↓
         [Repeat until done]
              ↓
          Final Answer
```

### 3. LangGraph Flow

```
START
  ↓
Node A (State Update)
  ↓
[Conditional Edge]
  ↓        ↓
Node B   Node C
  ↓        ↓
  [Merge]
     ↓
   Node D
     ↓
    END
```

### 4. RAG Flow

```
Query → Embed → Vector Search → Retrieve Docs
                                      ↓
                                 Format Context
                                      ↓
                                    Prompt
                                      ↓
                                     LLM
                                      ↓
                                Context-Aware Answer
```

## State Management

### LangChain Memory

```python
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

# State persists across calls
chain.predict("Hello")
chain.predict("What did I just say?")  # Remembers
```

### LangGraph State

```python
class State(TypedDict):
    messages: list[str]
    counter: int

def node_function(state: State) -> State:
    # Update state
    return {"counter": state["counter"] + 1}
```

## Extension Points

### Adding New Examples

1. Create module in appropriate directory
2. Follow the example template pattern
3. Add to `main.py` menu
4. Update README.md

### Adding Custom Tools

```python
# In custom_tools.py
def my_tool(input: str) -> str:
    """Tool description for the agent."""
    # Implementation
    return result

def create_my_tool() -> Tool:
    return Tool(
        name="my_tool",
        func=my_tool,
        description="When to use this tool"
    )
```

### Creating New Agent Types

```python
# In a new module
from langchain.agents import AgentExecutor, create_tool_calling_agent

def create_specialized_agent():
    tools = [...]
    llm = load_llm()
    prompt = create_specialized_prompt()
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)
```

## Testing Strategy

While this is an educational demo, you can add tests:

```python
# tests/test_chains.py
def test_basic_chain():
    from src.01_basics.chains import create_basic_chain
    
    chain = create_basic_chain()
    result = chain.invoke({"topic": "test"})
    
    assert result is not None
    assert len(result) > 0
```

## Performance Considerations

### Token Usage

- Use callbacks to track token consumption
- Implement token limits in chains
- Monitor costs with LangSmith

### Memory Management

- Use window memory for long conversations
- Implement summary memory for cost savings
- Clear memory when appropriate

### Concurrent Processing

```python
# Async for better performance
async def process_batch(items):
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)
```

## Security Best Practices

1. **API Keys**: Never commit `.env` file
2. **Input Validation**: Sanitize user inputs
3. **Rate Limiting**: Implement request throttling
4. **Error Handling**: Don't expose sensitive errors

## Deployment Considerations

### Production Checklist

- [ ] Environment variables configured
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Rate limiting added
- [ ] Monitoring set up (LangSmith)
- [ ] Token budgets established
- [ ] API key rotation plan
- [ ] Backup strategies

### Scaling Options

1. **Horizontal**: Multiple agent instances
2. **Vertical**: More powerful LLM models
3. **Caching**: Cache frequent queries
4. **Async**: Concurrent request handling

## Future Enhancements

Possible additions:
- [ ] Web UI (FastAPI + React)
- [ ] Database integration
- [ ] Custom embedding models
- [ ] More vector stores (Pinecone, Weaviate)
- [ ] Agent benchmarking
- [ ] A/B testing framework
- [ ] Production deployment guide

## Contributing

To contribute new examples:

1. Follow the existing code style
2. Include comprehensive comments
3. Add to the main menu
4. Update documentation
5. Test thoroughly

## Resources

- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [OpenAI API](https://platform.openai.com/docs)
- [LangSmith](https://smith.langchain.com/)

---

**Questions?** Check the inline code documentation or open an issue!

