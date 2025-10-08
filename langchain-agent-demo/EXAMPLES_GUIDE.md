# 📚 Examples Guide

Complete reference for all examples in the LangChain Agent Demo.

## Table of Contents

- [01 Basics](#01-basics)
- [02 Agents](#02-agents)
- [03 Memory](#03-memory)
- [04 LangGraph](#04-langgraph)
- [05 Multi-Agent](#05-multi-agent)
- [06 RAG](#06-rag)

---

## 01 Basics

### chains.py - Basic Chain Examples

**What you'll learn:**
- How to create simple chains
- LCEL (LangChain Expression Language) syntax
- Sequential chain composition
- Structured output parsing
- Multiple inputs handling

**Key Examples:**
1. **Simple Chain**: `prompt | llm | output_parser`
2. **Sequential Chain**: Multi-step processing
3. **Structured Output**: JSON parsing with Pydantic
4. **Multiple Inputs**: Complex prompt variables
5. **Chain Composition**: Advanced data flow

**Run:**
```bash
python -m src.01_basics.chains
```

**Best For:** Complete beginners to LangChain

---

### prompts.py - Prompt Engineering

**What you'll learn:**
- Prompt template creation
- Few-shot learning patterns
- System message best practices
- Prompt composition techniques
- Dynamic prompt generation

**Key Examples:**
1. **Basic Templates**: String interpolation
2. **Few-Shot Prompting**: Learning from examples
3. **System Messages**: Role-based behavior
4. **Prompt Composition**: Modular prompts
5. **Dynamic Prompts**: Runtime adaptation

**Run:**
```bash
python -m src.01_basics.prompts
```

**Best For:** Learning prompt engineering

---

### llm_examples.py - LLM Interactions

**What you'll learn:**
- Synchronous vs asynchronous calls
- Streaming responses
- Token usage tracking
- Temperature effects
- Batch processing

**Key Examples:**
1. **Basic Invocation**: Simple LLM calls
2. **Streaming**: Real-time token generation
3. **Token Tracking**: Monitor usage and costs
4. **Temperature Comparison**: Creativity control
5. **Async Operations**: Concurrent processing
6. **Batch Processing**: Multiple inputs
7. **Parameter Tuning**: Fine-tune behavior

**Run:**
```bash
python -m src.01_basics.llm_examples
```

**Best For:** Understanding LLM APIs

---

## 02 Agents

### custom_tools.py - Custom Tool Creation

**What you'll learn:**
- Function-based tools
- Class-based tools
- Structured inputs
- Error handling
- Tool factories

**Key Tools Included:**
- Calculator
- Time getter
- Word counter
- Web search (simulated)
- Wikipedia search
- File operations

**Run:**
```bash
python -m src.02_agents.custom_tools
```

**Best For:** Creating your own tools

---

### simple_agent.py - Basic Agents

**What you'll learn:**
- Agent fundamentals
- Tool selection
- Agent reasoning
- Multi-tool agents
- Research agents

**Key Examples:**
1. **Basic Agent**: Single tool usage
2. **Multi-Tool Agent**: Tool selection
3. **Research Agent**: Information gathering
4. **Agent Reasoning**: Thought process

**Run:**
```bash
python -m src.02_agents.simple_agent
```

**Best For:** First-time agent builders

---

### react_agent.py - ReAct Pattern Agents

**What you'll learn:**
- ReAct (Reasoning + Acting) pattern
- Complex workflows
- Tool chaining
- Error recovery
- Real-world scenarios

**Key Examples:**
1. **ReAct Pattern**: Understanding the loop
2. **Complex Workflow**: Multi-step tasks
3. **Error Handling**: Recovery strategies
4. **Real-World Scenario**: Practical application

**Run:**
```bash
python -m src.02_agents.react_agent
```

**Best For:** Advanced agent patterns

---

## 03 Memory

### conversation_memory.py - Conversation Context

**What you'll learn:**
- Buffer memory
- Window memory
- Summary memory
- Custom memory with LCEL
- Agent memory integration

**Key Examples:**
1. **Buffer Memory**: Full history
2. **Window Memory**: Recent messages only
3. **Summary Memory**: Compressed history
4. **Summary Buffer**: Hybrid approach
5. **Custom Memory Chain**: LCEL implementation
6. **Agent with Memory**: Stateful agents

**Run:**
```bash
python -m src.03_memory.conversation_memory
```

**Best For:** Building conversational AI

---

## 04 LangGraph

### simple_graph.py - Basic State Machines

**What you'll learn:**
- State graph concepts
- Nodes and edges
- State management
- LLM-powered nodes
- State accumulation
- Streaming execution

**Key Examples:**
1. **Basic Graph**: Linear workflow
2. **LLM Nodes**: Content generation pipeline
3. **State Accumulation**: Building up state
4. **Streaming Output**: Watch execution
5. **Graph Visualization**: Understanding structure

**Run:**
```bash
python -m src.04_langgraph.simple_graph
```

**Best For:** LangGraph introduction

---

### conditional_graph.py - Conditional Routing

**What you'll learn:**
- Conditional edges
- Dynamic routing
- Multi-way branching
- Cyclic workflows
- Parallel + conditional patterns

**Key Examples:**
1. **Simple Conditional**: Even/odd routing
2. **Multi-Way Routing**: Sentiment analysis
3. **Cyclic Workflow**: Iterative improvement
4. **Parallel & Conditional**: Complex workflows

**Run:**
```bash
python -m src.04_langgraph.conditional_graph
```

**Best For:** Advanced LangGraph patterns

---

## 05 Multi-Agent

### research_team.py - Multi-Agent Systems

**What you'll learn:**
- Specialized agents
- Agent coordination
- Supervisor patterns
- Collaborative workflows
- Parallel execution

**Key Examples:**
1. **Specialized Agents**: Research team
2. **Supervisor Pattern**: Task delegation
3. **Collaborative Agents**: Feedback loops
4. **Parallel Agents**: Simultaneous work

**Run:**
```bash
python -m src.05_multi_agent.research_team
```

**Best For:** Multi-agent architectures

---

## 06 RAG

### qa_system.py - Question Answering

**What you'll learn:**
- Document loading and chunking
- Vector embeddings
- Semantic search
- RAG pipeline construction
- Source citations
- Advanced retrieval (MMR)

**Key Examples:**
1. **Basic RAG**: Complete pipeline
2. **Custom RAG Chain**: LCEL implementation
3. **RAG with Sources**: Citation tracking
4. **Semantic Search**: Similarity exploration
5. **Advanced RAG**: MMR and filtering

**Run:**
```bash
python -m src.06_rag.qa_system
```

**Best For:** Knowledge-based systems

---

## Learning Paths

### Path 1: Complete Beginner (4-6 hours)

```
Day 1: Foundations
├── 01_basics/chains.py          (1 hour)
├── 01_basics/prompts.py         (1 hour)
└── 01_basics/llm_examples.py    (1 hour)

Day 2: Agents
├── 02_agents/simple_agent.py    (1 hour)
├── 03_memory/conversation_memory.py (1 hour)
└── 04_langgraph/simple_graph.py (1 hour)
```

### Path 2: Intermediate Developer (3-4 hours)

```
Session 1: Advanced Concepts
├── 02_agents/react_agent.py     (1 hour)
├── 04_langgraph/conditional_graph.py (1 hour)
└── 06_rag/qa_system.py          (1 hour)

Session 2: Complex Systems
└── 05_multi_agent/research_team.py (1 hour)
```

### Path 3: Production Focus (2-3 hours)

```
Focus on:
├── Error handling in react_agent.py
├── Token tracking in llm_examples.py
├── Memory optimization in conversation_memory.py
└── RAG performance in qa_system.py
```

---

## Example Difficulty Levels

### 🟢 Beginner
- `chains.py` - Basic chains
- `prompts.py` - Prompt templates
- `simple_agent.py` - First agents

### 🟡 Intermediate
- `llm_examples.py` - API patterns
- `conversation_memory.py` - State management
- `simple_graph.py` - LangGraph basics
- `qa_system.py` - RAG fundamentals

### 🔴 Advanced
- `react_agent.py` - Complex reasoning
- `conditional_graph.py` - Dynamic workflows
- `research_team.py` - Multi-agent systems

---

## Common Patterns Across Examples

### Pattern 1: Example Structure
```python
def example_N_name():
    """Example with description."""
    print_section("Title", "Description")
    print_step(1, "Step description")
    # Implementation
    print_response(result, "Title")
```

### Pattern 2: Error Handling
```python
try:
    result = agent.invoke(input)
except Exception as e:
    print_error(f"Error: {e}")
```

### Pattern 3: Configuration
```python
from config.settings import settings
llm = ChatOpenAI(model=settings.default_model)
```

---

## Tips for Each Example

### Chains
- Start with example 1, don't skip ahead
- Experiment with different prompts
- Try modifying the chain structure

### Prompts
- Copy examples and modify them
- Test different system messages
- Create your own few-shot examples

### LLM Examples
- Watch the streaming output
- Pay attention to token usage
- Try different temperatures

### Agents
- Read the verbose output carefully
- Understand the reasoning process
- Create custom tools

### Memory
- Test with multiple conversation turns
- Compare memory types
- Monitor memory size

### LangGraph
- Visualize the graph structure mentally
- Trace state changes
- Experiment with routing logic

### Multi-Agent
- Understand agent roles
- Watch the coordination
- Try different team structures

### RAG
- Use your own documents
- Experiment with chunk sizes
- Try different search parameters

---

## Next Steps After Completing Examples

1. **Build Your Own Agent**
   - Combine patterns you've learned
   - Use custom tools
   - Implement memory

2. **Create a Project**
   - Pick a real problem to solve
   - Design the architecture
   - Implement incrementally

3. **Explore Advanced Topics**
   - Production deployment
   - Monitoring and logging
   - Cost optimization
   - Performance tuning

4. **Contribute**
   - Add new examples
   - Improve documentation
   - Share your learnings

---

## Resources

- **LangChain Docs**: https://python.langchain.com/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **OpenAI Docs**: https://platform.openai.com/docs
- **LangSmith**: https://smith.langchain.com/

---

**Happy Learning! 🚀**

Got stuck? Check the inline code comments - they're very detailed!

