# 🎨 Visual Schemas & Architecture Diagrams

Beautiful visual representations of LangChain and LangGraph concepts.

## Table of Contents
- [System Architecture](#system-architecture)
- [Chain Patterns](#chain-patterns)
- [Agent Workflows](#agent-workflows)
- [LangGraph Patterns](#langgraph-patterns)
- [MCP Integration](#mcp-integration)
- [Monitoring Architecture](#monitoring-architecture)

---

## System Architecture

### Overall System

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  CLI Menu    │  │  Interactive │  │  Batch Executor      │  │
│  │  main.py     │  │  Prompts     │  │  run_all_examples.py │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│                     APPLICATION LAYER                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ Basics   │ │ Agents   │ │ Memory   │ │ Multi-Agent      │  │
│  │ Examples │ │ Examples │ │ Examples │ │ Examples         │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │LangGraph │ │   RAG    │ │   MCP    │ │ Monitoring       │  │
│  │ Examples │ │ Examples │ │Integration│ │ (LangSmith)      │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│                     UTILITIES LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │   Display    │  │   Helpers    │  │   Configuration    │   │
│  │   (Rich UI)  │  │   (Common)   │  │   (Settings)       │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│                    LANGCHAIN LAYER                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐ │
│  │ Chains   │ │ Agents   │ │ Memory   │ │ Vector Stores     │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐ │
│  │LangGraph │ │  Tools   │ │ Prompts  │ │ Output Parsers    │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘ │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│                      EXTERNAL SERVICES                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐ │
│  │ OpenAI   │ │ FAISS    │ │LangSmith │ │ MCP Servers       │ │
│  │ API      │ │ Vectors  │ │ Tracing  │ │ (Filesystem, DB)  │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Chain Patterns

### 1. Simple Chain (LCEL)

```
Input
  │
  ├─► PromptTemplate ─┐
  │                   ├─► Format Prompt ─► LLM ─► Output Parser ─► Result
  └─► Variables ──────┘

Example:
{"topic": "AI"} ─► "Tell me about {topic}" ─► LLM ─► Parse ─► "AI is..."
```

### 2. Sequential Chain

```
Input
  │
  ▼
┌────────────┐
│  Chain 1   │ → Intermediate Result 1
└────────────┘
  │
  ▼
┌────────────┐
│  Chain 2   │ → Intermediate Result 2
└────────────┘
  │
  ▼
┌────────────┐
│  Chain 3   │ → Final Result
└────────────┘
```

### 3. Parallel Chain

```
                    ┌─► Chain A ─► Result A ─┐
                    │                        │
Input ─► Distribute ├─► Chain B ─► Result B ─┼─► Combine ─► Final
                    │                        │
                    └─► Chain C ─► Result C ─┘
```

---

## Agent Workflows

### 1. Basic Agent Loop

```
┌─────────────────────────────────────────┐
│          START: User Query              │
└──────────────────┬──────────────────────┘
                   │
           ┌───────▼───────┐
           │  Agent (LLM)  │
           │    Thinks     │
           └───────┬───────┘
                   │
        ┌──────────┴──────────┐
        │   Decision Point    │
        │  Need Tool / Done?  │
        └──────────┬──────────┘
                   │
          ┌────────┴────────┐
          │                 │
        TOOL             ANSWER
          │                 │
    ┌─────▼─────┐          │
    │Execute    │          │
    │Tool       │          │
    └─────┬─────┘          │
          │                │
    ┌─────▼─────┐          │
    │Observation│          │
    └─────┬─────┘          │
          │                │
          └─────► LOOP ◄───┘
                   │
           ┌───────▼───────┐
           │ Final Answer  │
           └───────────────┘
```

### 2. ReAct Pattern

```
User Query
    │
    ▼
┌─────────────────┐
│   THOUGHT       │ "I need to search for information"
└────────┬────────┘
         │
    ┌────▼────┐
    │ ACTION  │ → Select Tool: "search"
    └────┬────┘
         │
    ┌────▼────┐
    │  INPUT  │ → Provide Arguments: "quantum computing"
    └────┬────┘
         │
  ┌──────▼──────┐
  │ OBSERVATION │ → Tool Result: "Quantum computing is..."
  └──────┬──────┘
         │
         ├─ Back to THOUGHT (if more info needed)
         │
         └─► ANSWER: "Based on my research..."
```

### 3. Multi-Agent System

```
                    ┌──────────────────┐
                    │   SUPERVISOR     │
                    │   (Coordinator)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
       │  RESEARCHER │ │  ANALYST │ │   WRITER   │
       │   Agent     │ │   Agent  │ │   Agent    │
       └──────┬──────┘ └────┬─────┘ └─────┬──────┘
              │              │              │
              │  Results     │  Results     │  Results
              └──────────────┼──────────────┘
                             │
                      ┌──────▼──────┐
                      │   COMBINE   │
                      │   Results   │
                      └──────┬──────┘
                             │
                      ┌──────▼──────┐
                      │Final Output │
                      └─────────────┘
```

---

## LangGraph Patterns

### 1. Linear Graph

```
START ─► Node A ─► Node B ─► Node C ─► END

State flows sequentially through each node.
```

### 2. Conditional Graph

```
              START
                │
                ▼
            ┌───────┐
            │Node A │
            └───┬───┘
                │
         ┌──────┴──────┐
         │  Condition  │
         └──────┬──────┘
                │
        ┌───────┴───────┐
        │               │
    ┌───▼───┐       ┌───▼───┐
    │Node B │       │Node C │
    └───┬───┘       └───┬───┘
        │               │
        └───────┬───────┘
                │
                ▼
               END
```

### 3. Cyclic Graph (Loop)

```
        START
          │
          ▼
    ┌──────────┐
    │  Process │
    └─────┬────┘
          │
    ┌─────▼─────┐
    │ Evaluate  │
    └─────┬─────┘
          │
    ┌─────┴─────┐
    │ Condition │
    └─────┬─────┘
          │
    ┌─────┴─────┐
    │           │
  LOOP        DONE
    │           │
    └─► Back    └─► END
```

### 4. Complex Graph

```
                     START
                       │
                ┌──────▼──────┐
                │   Node A    │
                └──────┬──────┘
                       │
           ┌───────────┴───────────┐
           │                       │
      ┌────▼────┐             ┌────▼────┐
      │ Node B  │             │ Node C  │
      └────┬────┘             └────┬────┘
           │                       │
           └───────────┬───────────┘
                       │
                ┌──────▼──────┐
                │   Node D    │
                │  (Merge)    │
                └──────┬──────┘
                       │
                      END
```

---

## MCP Integration

### MCP Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangChain Agent                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │                 Agent Core                         │    │
│  │  • Reasoning Logic                                 │    │
│  │  • Tool Selection                                  │    │
│  │  • Response Generation                             │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │                                         │
│  ┌────────────────┴───────────────────────────────────┐    │
│  │              MCP Tools Layer                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │    │
│  │  │ Read     │  │ Query    │  │ List Resources   │ │    │
│  │  │ Resource │  │ Database │  │                  │ │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────────────┘ │    │
│  └───────┼─────────────┼─────────────┼───────────────┘    │
└──────────┼─────────────┼─────────────┼────────────────────┘
           │             │             │
┌──────────┴─────────────┴─────────────┴────────────────────┐
│                    MCP Protocol Layer                      │
│  ┌────────────────────────────────────────────────────┐   │
│  │        Standard MCP Communication                  │   │
│  │   • JSON-RPC Messages                              │   │
│  │   • Resource URIs                                  │   │
│  │   • Tool Invocations                               │   │
│  └────────────────────────────────────────────────────┘   │
└──────────┬─────────────┬─────────────┬────────────────────┘
           │             │             │
┌──────────▼──────┐ ┌────▼──────┐ ┌───▼────────────────────┐
│  MCP Server 1   │ │ MCP       │ │ MCP Server 3           │
│  (Filesystem)   │ │ Server 2  │ │ (APIs)                 │
│                 │ │ (Database)│ │                        │
│  📁 Files       │ │ 💾 Data   │ │ 🌐 External Services   │
│  📄 Documents   │ │ 📊 Tables │ │ 🔌 Integrations        │
└─────────────────┘ └───────────┘ └────────────────────────┘
```

### MCP Resource Access Flow

```
1. Agent Needs Information
   │
   ▼
2. Select MCP Tool
   │   (e.g., "read_file")
   ▼
3. Format Request
   │   {uri: "file://docs/readme.md"}
   ▼
4. Send to MCP Server
   │
   ▼
5. Server Processes
   │   • Validates URI
   │   • Checks permissions
   │   • Retrieves resource
   ▼
6. Returns Response
   │   {content: "...", metadata: {...}}
   ▼
7. Agent Receives & Uses
   │
   ▼
8. Continue with Task
```

---

## Monitoring Architecture

### LangSmith Integration

```
┌────────────────────────────────────────────────────────────┐
│               Your LangChain Application                   │
│                                                            │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌─────────┐ │
│  │  Chains  │  │  Agents  │  │ LangGraph │  │   RAG   │ │
│  └─────┬────┘  └─────┬────┘  └─────┬─────┘  └────┬────┘ │
│        │             │              │             │      │
│        └─────────────┴──────────────┴─────────────┘      │
│                      │                                    │
│            ┌─────────▼──────────┐                        │
│            │  Tracing Callback  │                        │
│            │  • Captures calls  │                        │
│            │  • Records timing  │                        │
│            │  • Logs errors     │                        │
│            └─────────┬──────────┘                        │
└──────────────────────┼─────────────────────────────────┘
                       │
                       │ HTTPS
                       │
┌──────────────────────▼─────────────────────────────────┐
│                 LangSmith Cloud                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │              Trace Storage                       │ │
│  │  • All LLM calls                                 │ │
│  │  • Agent decisions                               │ │
│  │  • Tool executions                               │ │
│  │  • Performance metrics                           │ │
│  └──────────────────┬───────────────────────────────┘ │
│                     │                                  │
│  ┌──────────────────▼───────────────────────────────┐ │
│  │           Analytics Engine                       │ │
│  │  • Token counting                                │ │
│  │  • Cost calculation                              │ │
│  │  • Performance analysis                          │ │
│  │  • Error detection                               │ │
│  └──────────────────┬───────────────────────────────┘ │
│                     │                                  │
│  ┌──────────────────▼───────────────────────────────┐ │
│  │           Dashboard & UI                         │ │
│  │  📊 Real-time metrics                            │ │
│  │  📈 Cost analytics                               │ │
│  │  🐛 Debug traces                                 │ │
│  │  ⚠️  Alerts                                       │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### Monitoring Data Flow

```
   Application Event
         │
         ▼
┌────────────────────┐
│  Callback Handler  │
│  • on_llm_start    │
│  • on_llm_end      │
│  • on_tool_start   │
│  • on_tool_end     │
│  • on_error        │
└─────────┬──────────┘
          │
          ▼
  ┌───────────────┐
  │   Capture     │
  │   • Inputs    │
  │   • Outputs   │
  │   • Timing    │
  │   • Tokens    │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │   Send to     │
  │   LangSmith   │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │   Process &   │
  │   Analyze     │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │   Display in  │
  │   Dashboard   │
  └───────────────┘
```

---

## Data Flow: Complete RAG System

```
User Query: "What is LangChain?"
    │
    ▼
┌─────────────────┐
│  Embed Query    │ → Vector [0.2, 0.5, ...]
└────────┬────────┘
         │
    ┌────▼────┐
    │ Search  │
    │ Vector  │
    │  Store  │
    └────┬────┘
         │
    ┌────▼────────────────┐
    │ Retrieved Docs      │
    │ 1. "LangChain is..."│
    │ 2. "It provides..." │
    │ 3. "Key features..."│
    └────┬────────────────┘
         │
    ┌────▼─────────────────┐
    │ Format Context       │
    │ Combine docs into    │
    │ single context       │
    └────┬─────────────────┘
         │
    ┌────▼──────────────────┐
    │  Create Prompt        │
    │  "Based on context:   │
    │   [docs]              │
    │   Answer: [query]"    │
    └────┬──────────────────┘
         │
    ┌────▼────┐
    │   LLM   │ → Generate answer with context
    └────┬────┘
         │
    ┌────▼──────┐
    │  Response │ → "LangChain is a framework..."
    └───────────┘
```

---

## Memory Systems

### Buffer Memory

```
┌──────────────────────────────────┐
│      Conversation Buffer         │
│                                  │
│  [Human]: "Hi, I'm Alice"        │
│  [AI]: "Hello Alice!"            │
│  [Human]: "What's my name?"      │
│  [AI]: "Your name is Alice"      │
│                                  │
│  ✓ Stores everything             │
│  ✗ Can get very large            │
└──────────────────────────────────┘
```

### Window Memory

```
┌──────────────────────────────────┐
│      Conversation Window         │
│      (Last 2 messages)           │
│                                  │
│  [Forgotten older messages...]   │
│                                  │
│  [Human]: "What's my name?"      │
│  [AI]: "Your name is Alice"      │
│                                  │
│  ✓ Fixed size                    │
│  ✗ Loses older context           │
└──────────────────────────────────┘
```

### Summary Memory

```
┌──────────────────────────────────┐
│     Conversation Summary         │
│                                  │
│  Summary: "User Alice asked      │
│  about Python programming..."    │
│                                  │
│  [Recent]:                       │
│  [Human]: "Tell me more"         │
│  [AI]: "Python is..."            │
│                                  │
│  ✓ Compact                       │
│  ✓ Preserves key info            │
└──────────────────────────────────┘
```

---

**These schemas provide visual understanding of the system architecture and data flows!** 🎨

For interactive exploration, run the examples in the project.

