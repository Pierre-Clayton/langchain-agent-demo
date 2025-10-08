# 🤖 LangChain & LangGraph - Complete AI Agents Demo

**Complete Documentation - Everything you need to get started and master LangChain**

> 📚 This project is a comprehensive, pedagogical demonstration of LangChain and LangGraph, covering foundational concepts up to advanced multi-agent systems, with MCP integration and LangSmith monitoring.

---

## 📑 Table of Contents

1. [Introduction](#introduction)
2. [Full Installation](#full-installation)
3. [Configuration](#configuration)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Detailed Examples](#detailed-examples)
7. [Architecture](#architecture)
8. [MCP Integration](#mcp-integration)
9. [Monitoring with LangSmith](#monitoring-with-langsmith)
10. [Learning Path](#learning-path)
11. [Production](#production)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Introduction

### What is this project?

This demo is a **complete educational guide** to building AI applications with LangChain and LangGraph. It contains **60+ hands-on examples** covering:

- ✅ **Fundamentals**: Chains, prompts, LLM interaction patterns
- ✅ **Intelligent Agents**: Tools, ReAct pattern
- ✅ **Conversation Memory**: Context and history
- ✅ **State Machines**: Complex workflows with LangGraph
- ✅ **Multi-Agent Systems**: Coordination and collaboration
- ✅ **RAG**: Semantic search and document Q&A
- ✅ **MCP**: Real production servers (Fetch & Canva)
- ✅ **Monitoring**: Observability and debugging with LangSmith

### Why use this project?

- 📖 **Pedagogical**: Every example is documented and explained
- 🎯 **Progressive**: From beginner to expert
- 💻 **Practical**: Ready-to-run code
- 🏭 **Production**: Best practices included
- 🎨 **Interactive**: Beautiful CLI interface

---

## 🚀 Full Installation

### Prerequisites

Before you begin, make sure you have:

- **Python 3.10 or higher** ([Download Python](https://www.python.org/downloads/))
- **An OpenAI API key** ([Get a key](https://platform.openai.com/api-keys))
- **Git** (optional, to clone the project)

### Step 1: Prepare the environment

#### Option A: Automated Setup (Recommended)

```bash
# 1. Navigate into the project folder
cd langchain-agent-demo

# 2. Run the setup script
python setup.py
```

The script will:
- ✅ Verify your Python version
- ✅ Create the `.env` file
- ✅ Create required directories
- ✅ Offer to install dependencies

#### Option B: Manual Setup

```bash
# 1. Navigate into the project folder
cd langchain-agent-demo

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create the configuration file
cp .env.example .env
```

### Step 2: Verify the installation

```bash
# Verify installation
python -c "import langchain; import langgraph; print('✅ Installation successful!')"
```

If you see "✅ Installation successful!" you're good to go!

---

## ⚙️ Configuration

### Basic Configuration (Required)

#### 1. Set your OpenAI API Key

Open the `.env` file and add your API key:

```bash
# .env
OPENAI_API_KEY=sk-your-api-key-here
```

**How to get an OpenAI API key:**
1. Go to https://platform.openai.com/
2. Create an account or sign in
3. Go to "API Keys"
4. Click "Create new secret key"
5. Copy the key and paste it into `.env`

#### 2. Choose the Model (Optional)

By default, the project uses `gpt-4o-mini`. You can change it:

```bash
# .env
DEFAULT_MODEL=gpt-4o-mini  # Cost-effective
# Or use:
# DEFAULT_MODEL=gpt-4o      # More powerful, higher cost
```

### Advanced Configuration (Optional)

#### 1. LangSmith (Monitoring)

If you want to enable monitoring and tracing:

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=langchain-agent-demo
```

**How to get LangSmith:**
1. Go to https://smith.langchain.com/
2. Create a free account
3. Create a project
4. Copy your API key

#### 2. Other LLM Providers (Optional)

```bash
# .env
ANTHROPIC_API_KEY=your-anthropic-key
COHERE_API_KEY=your-cohere-key
```

### Validate Configuration

```bash
# Test that configuration is valid
python -c "from config.settings import validate_api_keys; validate_api_keys()"
```

---

## 🎮 Quick Start

### Method 1: Interactive Interface (Recommended)

The interactive interface is the **easiest** way to explore all examples:

```bash
python main.py
```

You’ll see a clean menu:

```
🤖 LangChain & LangGraph Educational Demo

📚 Example Categories
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
No.  Category                Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1    Basic Chains           Learn fundamental concepts
2    Prompt Engineering     Master prompt templates
3    LLM Interactions       Explore LLM patterns
...
```

**Navigation:**
- Type the **number** of the example you want to try
- Type **help** for a quick guide
- Type **about** for more info
- Type **0** to exit

### Method 2: Run All Examples

To run all examples sequentially:

```bash
python run_all_examples.py
```

⚠️ **Note**: This can take time and consume API credits!

### Method 3: Run a Specific Example

Run a specific module directly:

```bash
# Example: Basic chains
python -m src.01_basics.chains

# Example: Simple agent
python -m src.02_agents.simple_agent

# Example: LangGraph
python -m src.04_langgraph.simple_graph

# Example: Multi-agent system
python -m src.05_multi_agent.research_team

# Example: RAG (document Q&A)
python -m src.06_rag.qa_system

# Example: MCP Integration
python -m src.07_mcp.mcp_integration

# Example: LangSmith Monitoring
python -m src.08_monitoring.langsmith_monitoring
```

### Quick Test

Quickly verify that everything works:

```bash
# Simple test
python -c "from langchain_openai import ChatOpenAI; from config.settings import settings; llm = ChatOpenAI(model=settings.default_model); print(llm.invoke('Hello!').content)"
```

If you see a response, you’re all set! 🎉

---

## 📁 Project Structure

### Overview

```
langchain-agent-demo/
│
├── 📄 README.md                    # This file - Complete documentation
├── 📦 requirements.txt             # Python dependencies
├── ⚙️  .env.example                # Configuration template
├── 🚀 main.py                      # Interactive interface (ENTRY POINT)
├── 🏃 run_all_examples.py          # Run all examples
├── 🔧 setup.py                     # Setup script
│
├── 📂 config/                      # Configuration
│   ├── __init__.py
│   └── settings.py                 # Environment variables management
│
├── 📂 src/                         # Source code (8 modules)
│   ├── __init__.py
│   │
│   ├── 01_basics/                  # MODULE 1: Fundamentals
│   │   ├── chains.py               # Basic chains (7 examples)
│   │   ├── prompts.py              # Prompt engineering (5 examples)
│   │   └── llm_examples.py         # LLM interactions (7 examples)
│   │
│   ├── 02_agents/                  # MODULE 2: Agents
│   │   ├── custom_tools.py         # Custom tools
│   │   ├── simple_agent.py         # Basic agents (4 examples)
│   │   └── react_agent.py          # ReAct pattern (4 examples)
│   │
│   ├── 03_memory/                  # MODULE 3: Memory
│   │   └── conversation_memory.py  # Conversation memory (6 examples)
│   │
│   ├── 04_langgraph/               # MODULE 4: State Machines
│   │   ├── simple_graph.py         # Simple graphs (5 examples)
│   │   └── conditional_graph.py    # Conditional routing (4 examples)
│   │
│   ├── 05_multi_agent/             # MODULE 5: Multi-Agents
│   │   └── research_team.py        # Team workflows (4 examples)
│   │
│   ├── 06_rag/                     # MODULE 6: RAG
│   │   └── qa_system.py            # Q&A system (5 examples)
│   │
│   ├── 07_mcp/                     # MODULE 7: MCP Integration
│   │   └── mcp_integration.py      # MCP protocol (5 examples)
│   │
│   ├── 08_monitoring/              # MODULE 8: Monitoring
│   │   └── langsmith_monitoring.py # LangSmith (6 examples)
│   │
│   └── utils/                      # Utilities
│       ├── display.py              # Rich terminal UI
│       └── helpers.py              # Common helpers
│
└── 📂 examples/                    # Sample data
    └── sample_documents/
        └── sample.txt              # Example doc for RAG
```

### Module Breakdown

| Module | Files | Examples | Description |
|--------|-------|----------|-------------|
| **01_basics** | 3 | 19 | Basic chains, prompts, and LLM interactions |
| **02_agents** | 3 | 8 | Agents with tools, ReAct pattern |
| **03_memory** | 1 | 6 | Conversation memory management |
| **04_langgraph** | 2 | 9 | State machines and workflows |
| **05_multi_agent** | 1 | 4 | Coordinated multi-agent systems |
| **06_rag** | 1 | 5 | Semantic search and Q&A |
| **07_mcp** | 1 | 5 | Model Context Protocol integration |
| **08_monitoring** | 1 | 6 | Monitoring and observability |
| **TOTAL** | **13** | **62** | **Complete, runnable examples** |

---

## 📖 Detailed Examples

### MODULE 1: Fundamentals (01_basics/)

#### 📝 chains.py - Basic Chains

**What you’ll learn:**
- How to build a simple chain
- LCEL (LangChain Expression Language) syntax
- Sequential chain composition
- Structured output parsing (JSON)
- Handling multiple inputs

**Included examples:**

1. **Simple Chain**: `prompt | llm | output_parser`
   ```python
   prompt = ChatPromptTemplate.from_template("Tell a joke about {topic}")
   chain = prompt | llm | StrOutputParser()
   response = chain.invoke({"topic": "programming"})
   ```

2. **Sequential Chain**: Multi-step pipeline
3. **Structured Output**: Parse JSON with Pydantic
4. **Multiple Inputs**: Multi-variable prompt
5. **Chain Composition**: Complex flows with RunnablePassthrough

**Run:**
```bash
python -m src.01_basics.chains
```

---

#### 💬 prompts.py - Prompt Engineering

**What you’ll learn:**
- Creating prompt templates
- Few-shot learning
- System messages and roles
- Prompt composition
- Dynamic prompts

**Included examples:**

1. **Basic Templates**: Variables and formatting
2. **Few-Shot Prompting**: Teach by example
3. **System Messages**: Control model behavior
4. **Prompt Composition**: Modular prompts
5. **Dynamic Prompts**: Runtime adaptation

**Run:**
```bash
python -m src.01_basics.prompts
```

---

#### 🤖 llm_examples.py - LLM Interactions

**What you’ll learn:**
- Sync vs async calls
- Streaming real-time responses
- Token and cost tracking
- Temperature effects
- Batch processing

**Included examples:**

1. **Basic Invocation**: Simple LLM call
2. **Streaming**: Token-by-token generation
3. **Cost Tracking**: Monitor tokens and costs
4. **Temperature Comparison**: Creativity control
5. **Async Operations**: Concurrent processing
6. **Batch Processing**: Efficient multi-input
7. **Parameter Tuning**: Fine-tune behavior

**Run:**
```bash
python -m src.01_basics.llm_examples
```

---

### MODULE 2: Agents (02_agents/)

#### 🛠️ custom_tools.py - Custom Tools

**What you’ll learn:**
- Function-based tools
- Class-based tools
- Structured inputs
- Error handling in tools

**Tools included:**
- ✅ Calculator (math expressions)
- ✅ Clock (current time)
- ✅ Word counter
- ✅ Web search (simulated)
- ✅ Wikipedia search (real API)
- ✅ File operations

**Run:**
```bash
python -m src.02_agents.custom_tools
```

---

#### 🤖 simple_agent.py - Basic Agents

**What you’ll learn:**
- What is an agent and how it works
- Tool selection and usage
- Agent reasoning process
- Multi-tool agents

**Included examples:**

1. **Basic Agent**: Single tool (calculator)
2. **Multi-Tool Agent**: Agent decides which tool to use
3. **Research Agent**: Information lookup
4. **Agent Reasoning**: View the thinking process

**Run:**
```bash
python -m src.02_agents.simple_agent
```

---

#### 🔄 react_agent.py - ReAct Pattern

**What you’ll learn:**
- ReAct pattern (Reasoning + Acting)
- Complex multi-step workflows
- Tool chaining
- Error recovery

**Included examples:**

1. **ReAct Pattern**: Thought → Action → Observation loop
2. **Complex Workflow**: Multi-step tasks
3. **Error Handling**: Recovery and retry
4. **Real-World Scenario**: Practical application

**Run:**
```bash
python -m src.02_agents.react_agent
```

---

### MODULE 3: Memory (03_memory/)

#### 🧠 conversation_memory.py - Conversation Memory

**What you’ll learn:**
- Different memory types
- Conversation context management
- Memory with LCEL
- Memory + agents integration

**Memory Types:**

1. **Buffer Memory**: Full history
   - ✅ Stores all messages
   - ❌ Can grow large

2. **Window Memory**: Last N messages
   - ✅ Fixed size
   - ❌ Forgets older context

3. **Summary Memory**: Compressed summary
   - ✅ Compact
   - ✅ Preserves key info

4. **Summary Buffer Memory**: Hybrid
   - ✅ Recent + summary of older

5. **Custom Memory**: With LCEL

6. **Agent with Memory**: An agent that remembers

**Run:**
```bash
python -m src.03_memory.conversation_memory
```

---

### MODULE 4: LangGraph (04_langgraph/)

#### 📊 simple_graph.py - Simple Graphs

**What you’ll learn:**
- What is a state graph
- Nodes and edges
- Managing state between nodes
- LLM-powered nodes

**Included examples:**

1. **Linear Graph**: START → A → B → C → END
2. **LLM Nodes**: Content generation pipeline
3. **State Accumulation**: Build up state across nodes
4. **Streaming Execution**: See each step in real-time
5. **Graph Visualization**: Understand structure

**Diagram:**
```
START
  ↓
Node A (Process input)
  ↓
Node B (Transform)
  ↓
Node C (Finalize)
  ↓
END
```

**Run:**
```bash
python -m src.04_langgraph.simple_graph
```

---

#### 🔀 conditional_graph.py - Conditional Routing

**What you’ll learn:**
- Conditional edges
- Dynamic routing based on state
- Cyclic workflows (loops)
- Combining parallel + conditional

**Included examples:**

1. **Simple Conditional**: Even/odd routing
2. **Multi-Way Routing**: Sentiment analysis → 3 paths
3. **Cyclic Workflow**: Iterative improvement
4. **Parallel + Conditional**: Complex workflow

**Diagram:**
```
        START
          ↓
    ┌─────────┐
    │Decision │
    └─────────┘
          ↓
    ┌─────┴─────┐
    ↓           ↓
  Path A      Path B
    ↓           ↓
    └─────┬─────┘
          ↓
         END
```

**Run:**
```bash
python -m src.04_langgraph.conditional_graph
```

---

### MODULE 5: Multi-Agent (05_multi_agent/)

#### 👥 research_team.py - Research Team

**What you’ll learn:**
- Coordinating specialized agents
- Supervisor pattern
- Collaborative workflows
- Parallel agent execution

**Included examples:**

1. **Specialized Agents**: Researcher → Analyst → Writer
2. **Supervisor Pattern**: One supervisor coordinates agents
3. **Collaborative Agents**: Feedback and revision
4. **Parallel Agents**: Multiple agents working simultaneously

**Multi-Agent Architecture:**
```
         Supervisor
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
Researcher  Analyst   Writer
    ↓         ↓         ↓
    └─────────┴─────────┘
              ↓
        Final Result
```

**Run:**
```bash
python -m src.05_multi_agent.research_team
```

---

### MODULE 6: RAG (06_rag/)

#### 📄 qa_system.py - Question Answering System

**What you’ll learn:**
- Loading and chunking documents
- Vector embeddings
- Semantic search
- Building a complete RAG pipeline

**Included examples:**

1. **Basic RAG**: End-to-end pipeline
2. **Custom RAG Chain**: With LCEL
3. **RAG with Sources**: Source citations
4. **Semantic Search**: Similarity exploration
5. **Advanced RAG**: Maximum Marginal Relevance (MMR)

**RAG Architecture:**
```
Question
    ↓
Embeddings
    ↓
Vector Search
    ↓
Relevant Documents
    ↓
Context + Prompt
    ↓
LLM
    ↓
Answer with Context
```

**Run:**
```bash
python -m src.06_rag.qa_system
```

---

### MODULE 7: MCP (07_mcp/)

#### 🔌 mcp_integration.py - Real MCP Server Integration

**What is MCP?**

The **Model Context Protocol (MCP)** is an open standard that enables LLMs to connect to data sources and tools securely.

**Real Servers Integrated:**

1. **🌐 Fetch MCP Server**
   - Retrieve and process web content
   - Convert HTML to markdown
   - Repository: [modelcontextprotocol/servers/fetch](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)

2. **🎨 Canva Dev MCP Server**
   - AI-powered Canva development assistance
   - Canva app development guidance
   - Documentation: [canva.dev/mcp-server](https://www.canva.dev/docs/apps/mcp-server/)

**Key Concepts:**
- **Resources**: Read-only data access
- **Tools**: Actions via JSON-RPC
- **Servers**: Real production MCP servers
- **Protocol**: Stdio-based communication

**What you'll learn:**
- Connecting to real MCP servers
- Using Fetch server for web content
- Integrating Canva development tools
- Multi-server coordination
- Production best practices

**Included examples:**

1. **MCP Basics**: Connect to real servers (Fetch & Canva)
2. **Fetch Tool in Agent**: Web scraping capabilities
3. **Multiple MCP Servers**: Combine Fetch + Canva
4. **Best Practices**: Security and production patterns
5. **Real-World Scenario**: Research assistant with web access

**MCP Architecture:**
```
LangChain Agent
      ↓
MCP Tools
      ↓
MCP Protocol (JSON-RPC)
      ↓
  ┌───┴────┐
  ↓        ↓
Fetch    Canva
Server   Server
  ↓        ↓
Web      Canva
Content   API
```

**Prerequisites:**
```bash
# Fetch server
pip install mcp-server-fetch

# Canva server (requires Node.js and npm)
node --version  # v20+
npm --version
```

**Run:**
```bash
python -m src.07_mcp.mcp_integration
```

📖 **Full Setup Guide**: See [MCP_SERVERS_SETUP.md](MCP_SERVERS_SETUP.md)

---

### MODULE 8: Monitoring (08_monitoring/)

#### 📈 langsmith_monitoring.py - LangSmith Monitoring

**What is LangSmith?**

**LangSmith** is LangChain’s observability platform to:
- 🔍 Trace all LLM calls
- 📊 Analyze performance
- 💰 Track costs
- 🐛 Debug issues
- 📈 Improve quality

**What you’ll learn:**
- Tracing setup
- Custom callbacks
- Cost and token tracking
- Debugging failures
- Performance metrics

**Included examples:**

1. **Enable Tracing**: LangSmith setup
2. **Monitor Agent Execution**: Track full execution
3. **Track Costs**: Tokens and costs in real-time
4. **Debug Failures**: Capture and analyze
5. **Performance Metrics**: Latency, throughput
6. **Production Monitoring**: Full checklist

**LangSmith Dashboard:**
```
📊 Real-time metrics
💰 Cost analytics
🐛 Debug traces
⚠️  Alerts
📈 Historical trends
```

**Run:**
```bash
python -m src.08_monitoring.langsmith_monitoring
```

---

## 🏗️ Architecture

### Global Architecture

```
┌─────────────────────────────────────────────┐
│               INTERFACE LAYER               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   CLI    │  │Interactive│  │  Batch   │  │
│  │ main.py  │  │  Prompts  │  │ Runner   │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│             APPLICATION LAYER               │
│  ┌─────┐ ┌─────┐ ┌──────┐ ┌─────────────┐  │
│  │01-03│ │04-05│ │  06  │ │   07-08     │  │
│  │Basic│ │Graph│ │ RAG  │ │ MCP/Monitor │  │
│  └─────┘ └─────┘ └──────┘ └─────────────┘  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│              UTILITIES LAYER                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Display  │  │ Helpers  │  │  Config  │  │
│  │  (Rich)  │  │ (Common) │  │(Settings)│  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│              LANGCHAIN LAYER                │
│  ┌──────┐ ┌──────┐ ┌───────┐ ┌──────────┐  │
│  │Chains│ │Agents│ │Memory │ │  Graphs  │  │
│  └──────┘ └──────┘ └───────┘ └──────────┘  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│              EXTERNAL SERVICES              │
│  ┌────────┐ ┌──────┐ ┌──────────┐ ┌──────┐ │
│  │ OpenAI │ │FAISS │ │LangSmith │ │ MCP  │ │
│  └────────┘ └──────┘ └──────────┘ └──────┘ │
└─────────────────────────────────────────────┘
```

### Simple Chain Flow

```
User Input
        ↓
  Prompt Template
        ↓
   Format with Variables
        ↓
       LLM (OpenAI)
        ↓
    Output Parser
        ↓
       Result
```

### Agent Flow

```
User Question
        ↓
   ┌────────────┐
   │   Agent    │
   │ (Thinking) │
   └─────┬──────┘
         │
    ┌────┴────┐
    │Decision │
    └────┬────┘
         │
    ┌────┴────────┐
    │             │
   TOOL        ANSWER
    │             │
┌───┴───┐         │
│Execute│         │
└───┬───┘         │
    │             │
┌───┴────┐        │
│Observe │        │
└───┬────┘        │
    │             │
    └──► LOOP ◄───┘
```

### LangGraph Flow

```
     START
       ↓
   Node A (Initial State)
       ↓
     Condition?
       ↓
   ┌───┴───┐
   │       │
 Node B  Node C
   │       │
   └───┬───┘
       ↓
   Node D (Merge)
       ↓
      END
```

### RAG Flow

```
Question
    ↓
Question Embedding
    ↓
Vector Database Search
    ↓
Top-K Similar Documents
    ↓
Context Formatting
    ↓
Prompt = Context + Question
    ↓
LLM Generates Answer
    ↓
Answer with Sources
```

---

## 🔌 MCP Integration

### What is MCP?

The **Model Context Protocol** is a standard protocol that lets LLMs access:
- 📁 **File systems**
- 💾 **Databases**
- 🌐 **External APIs**
- 🔧 **Custom services**

### MCP Architecture

```
┌────────────────────────────────────┐
│           LangChain Agent          │
│   • Reasoning logic                │
│   • Tool selection                 │
└─────────────┬──────────────────────┘
              │
┌─────────────┴──────────────────────┐
│          MCP Tools Layer           │
│  ┌──────┐  ┌──────┐  ┌──────────┐ │
│  │ Read │  │Query │  │  List    │ │
│  └──────┘  └──────┘  └──────────┘ │
└─────────────┬──────────────────────┘
              │
┌─────────────┴──────────────────────┐
│          MCP Protocol              │
│  • JSON-RPC messages               │
│  • Resource URIs                   │
└─────────────┬──────────────────────┘
              │
┌─────────────┴──────────────────────┐
│          MCP Servers               │
│  ┌────────┐ ┌────────┐ ┌────────┐ │
│  │ Files  │ │Database│ │  APIs  │ │
│  └────────┘ └────────┘ └────────┘ │
└────────────────────────────────────┘
```

### Basic Usage

```python
# 1. Connect to the MCP server
mcp_server = SimulatedMCPServer("filesystem")

# 2. Create a LangChain tool
def read_file(path: str) -> str:
    return mcp_server.call_tool("read_file", {"path": path})

tool = Tool(
    name="read_file",
    func=read_file,
    description="Read a file via MCP"
)

# 3. Use in an agent
agent = create_agent_with_tools([tool])
```

### Key Concepts

**Resources**
- Read-only access
- Standard URIs: `file://`, `db://`, `api://`
- Includes metadata

**Tools**
- Actions that change state or compute
- Structured arguments
- Typed results

**Servers**
- One server = multiple resources/tools
- JSON-RPC communication
- Isolation and security

---

## 📈 Monitoring with LangSmith

### LangSmith Setup

#### 1. Get an API Key

1. Go to https://smith.langchain.com/
2. Create a (free) account
3. Create a project
4. Copy your API key

#### 2. Configuration

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key_here
LANGCHAIN_PROJECT=my-project
```

#### 3. Verification

```python
import os
print("Tracing enabled:", os.getenv("LANGCHAIN_TRACING_V2"))
```

### Features

**1. Automatic Tracing**
- All LLM calls are traced
- All agent executions
- All tool usages

**2. Dashboard**
- Real-time metrics
- Trace visualization
- Cost analytics

**3. Debugging**
- Replay requests
- Inspect prompts
- Analyze errors

**4. Optimization**
- Identify bottlenecks
- Reduce costs
- Improve prompts

### Usage

```python
# Tracing is automatic once configured
llm = ChatOpenAI()
response = llm.invoke("Hello")
# ✅ Automatically traced in LangSmith!

# For custom cost tracking
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = chain.invoke(input)
    print(f"Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost:.4f}")
```

---

## 🎓 Learning Path

### For Beginners (6–8 hours)

**Day 1: Fundamentals**
```
1️⃣ chains.py         (1h)  → Understand chains
2️⃣ prompts.py        (1h)  → Master prompts
3️⃣ llm_examples.py   (1h)  → LLM interactions
```

**Day 2: Agents & State**
```
4️⃣ simple_agent.py   (1h)  → First agent
5️⃣ conversation_memory.py (1h) → Add memory
6️⃣ simple_graph.py   (2h)  → State machines
```

**Goal:** Understand the basics and build a simple agent

---

### For Intermediates (4–6 hours)

**Session 1: Advanced Concepts**
```
7️⃣ react_agent.py         (1.5h) → Advanced reasoning
8️⃣ conditional_graph.py   (1.5h) → Complex workflows
9️⃣ qa_system.py           (2h)   → RAG system
```

**Session 2: Complex Systems**
```
🔟 research_team.py       (2h)   → Multi-agents
```

**Goal:** Master advanced patterns

---

### For Advanced Users (3–4 hours)

**Production & Integration**
```
1️⃣1️⃣ mcp_integration.py       (1.5h) → MCP integration
1️⃣2️⃣ langsmith_monitoring.py  (1.5h) → Monitoring
```

**Personal Project** (2h)
- Build your own system
- Integrate your data
- Deploy to production

**Goal:** Create production-ready systems

---

### Learning Tips

✅ **DO**
- Start with the basics (simple chains)
- Run each example
- Read code comments
- Modify examples to experiment
- Test with your own data

❌ **DON'T**
- Skip the fundamentals
- Copy-paste without understanding
- Ignore verbose outputs
- Neglect error handling
- Forget cost monitoring

---

## 🏭 Production

### Production Checklist

#### Security
- [ ] Secure API keys (never in code!)
- [ ] Validate user inputs
- [ ] PII filtering
- [ ] Configure MCP permissions
- [ ] Enable audit logs

#### Performance
- [ ] Async for concurrent operations
- [ ] Cache frequent requests
- [ ] DB connection pooling
- [ ] Rate limiting
- [ ] Appropriate timeouts

#### Observability
- [ ] LangSmith tracing enabled
- [ ] Metrics collected
- [ ] Alerts configured
- [ ] Dashboards built
- [ ] Centralized logs

#### Costs
- [ ] Budgets defined
- [ ] Cost alerts
- [ ] Prompt optimization
- [ ] Use appropriate models
- [ ] Cache enabled

#### Reliability
- [ ] Retry logic implemented
- [ ] Circuit breakers configured
- [ ] Health checks in place
- [ ] Fallback mechanisms
- [ ] Load testing executed

### Production Configuration

```python
# config/production.py

import os
from pathlib import Path

class ProductionConfig:
    # API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_MODEL = "gpt-4o-mini"  # More cost-effective
    
    # Limits
    MAX_TOKENS = 2000
    TIMEOUT = 30
    MAX_RETRIES = 3
    
    # Monitoring
    LANGSMITH_ENABLED = True
    LANGSMITH_SAMPLING = 1.0  # 100% in prod
    
    # Cache
    CACHE_ENABLED = True
    CACHE_TTL = 3600
    
    # Rate Limiting
    RATE_LIMIT = "100/minute"
    
    # Alerts
    ERROR_THRESHOLD = 0.05  # 5%
    COST_ALERT_THRESHOLD = 10.0  # $10/hour
```

### Production Patterns

**1. Retry with Backoff**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_llm_with_retry(prompt):
    return llm.invoke(prompt)
```

**2. Circuit Breaker**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failure_count = 0
        self.threshold = failure_threshold
        self.is_open = False
    
    def call(self, func):
        if self.is_open:
            raise Exception("Circuit breaker is open")
        
        try:
            result = func()
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.threshold:
                self.is_open = True
            raise e
```

**3. Full Monitoring**
```python
from langsmith import Client
import time

class ProductionMonitoring:
    def __init__(self):
        self.client = Client()
        
    def execute_with_monitoring(self, agent, input):
        start = time.time()
        
        try:
            # Execute
            result = agent.invoke(input)
            
            # Log success
            self.log_success(
                duration=time.time() - start,
                tokens=result.get('tokens', 0)
            )
            
            return result
            
        except Exception as e:
            # Log error
            self.log_error(
                error=str(e),
                duration=time.time() - start
            )
            raise
```

---

## 🔧 Troubleshooting

### Common Issues

#### ❌ "ModuleNotFoundError: No module named 'langchain'"

**Cause:** Dependencies not installed

**Solution:**
```bash
pip install -r requirements.txt
```

---

#### ❌ "OPENAI_API_KEY not set"

**Cause:** API key not configured

**Solution:**
1. Create the `.env` file
2. Add: `OPENAI_API_KEY=your-key`
3. Verify: `cat .env`

---

#### ❌ "RateLimitError: You exceeded your current quota"

**Cause:** API quota exceeded

**Solution:**
1. Check usage: https://platform.openai.com/usage
2. Add credits
3. Or use `gpt-4o-mini` (cheaper)

---

#### ❌ "ImportError: cannot import name 'ChatOpenAI'"

**Cause:** Incompatible version

**Solution:**
```bash
pip install --upgrade langchain langchain-openai
```

---

#### ❌ Example hangs or is slow

**Cause:** Possible network or API issue

**Solution:**
- Check your internet connection
- Try a shorter timeout
- Check OpenAI status: https://status.openai.com/

---

#### ❌ "UnicodeDecodeError" when loading documents

**Cause:** File encoding

**Solution:**
```python
# Use UTF-8 encoding
with open(file, 'r', encoding='utf-8') as f:
    content = f.read()
```

---

### Advanced Debugging

**Enable detailed logs:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**See API requests:**
```python
# Turn on verbose in agents
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # ✅ See all steps
)
```

**Test API connectivity:**
```python
from langchain_openai import ChatOpenAI

try:
    llm = ChatOpenAI()
    response = llm.invoke("test")
    print("✅ API works!")
except Exception as e:
    print(f"❌ Error: {e}")
```

---

## 📊 Metrics & Performance

### Estimated Costs

**By module (approximate with gpt-4o-mini):**

| Module | Examples | Est. Cost | Time |
|--------|----------|-----------|------|
| 01_basics | 19 | ~$0.50 | 30 min |
| 02_agents | 8 | ~$0.80 | 20 min |
| 03_memory | 6 | ~$0.40 | 15 min |
| 04_langgraph | 9 | ~$0.60 | 25 min |
| 05_multi_agent | 4 | ~$1.00 | 20 min |
| 06_rag | 5 | ~$0.70 | 20 min |
| 07_mcp | 5 | ~$0.50 | 20 min |
| 08_monitoring | 6 | ~$0.30 | 15 min |
| **TOTAL** | **62** | **~$4.80** | **~3h** |

💡 **Tip:** Use `gpt-4o-mini` for learning (10x cheaper than GPT-4)

---

## 🎯 Quick Summary

### Start in 5 Minutes

```bash
# 1. Install
cd langchain-agent-demo
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=your-key

# 3. Launch
python main.py

# 4. Pick an example and explore!
```

### Code Structure

```
📦 8 Modules
📝 62 Examples
🎓 3 Levels (Beginner, Intermediate, Advanced)
⏱️  ~3 hours of content
💰 ~$5 API costs (gpt-4o-mini)
```

### Next Steps

1. ✅ **Install** the project
2. ✅ **Configure** your API key
3. ✅ **Run** `python main.py`
4. ✅ **Start** with the basics (option 1)
5. ✅ **Progress** to advanced concepts
6. ✅ **Build** your own project!

---

## 📞 Support & Resources

### Official Documentation

- **LangChain**: https://python.langchain.com/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **OpenAI API**: https://platform.openai.com/docs
- **LangSmith**: https://docs.smith.langchain.com/
- **MCP**: https://modelcontextprotocol.io/

### Community

- **LangChain Discord**: https://discord.gg/langchain
- **GitHub**: https://github.com/langchain-ai/langchain

### In this Project

- All examples contain detailed comments
- Each module includes inline documentation
- Errors include helpful messages
- The CLI includes a `help` option

---

## 🎉 Conclusion

You now have **everything you need** to:

✅ Understand LangChain and LangGraph
✅ Build intelligent agents
✅ Create multi-agent systems
✅ Integrate data with RAG and MCP
✅ Monitor and optimize your applications
✅ Deploy to production

**Get started now:**

```bash
python main.py
```

**Happy learning and coding! 🚀🤖**

---

*Last updated: 2025*
*Version: 2.0*
*License: MIT*
