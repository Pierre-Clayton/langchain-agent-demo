# 🔌📈 MCP Integration & Monitoring Guide

Complete guide for Model Context Protocol integration and LangSmith monitoring.

## 📑 Table of Contents

1. [MCP Integration](#mcp-integration)
2. [LangSmith Monitoring](#langsmith-monitoring)
3. [Best Practices](#best-practices)
4. [Production Deployment](#production-deployment)

---

## 🔌 MCP Integration

### What is MCP?

**Model Context Protocol (MCP)** is an open standard that enables LLMs to securely connect to data sources and tools. It provides a universal interface for:
- Reading data (resources)
- Taking actions (tools)
- Using templates (prompts)

### Real MCP Servers Used in This Demo

This project integrates with **two real production MCP servers**:

1. **🌐 Fetch MCP Server**
   - **Purpose**: Retrieve and process web content
   - **Features**: Converts HTML to markdown, handles various web formats
   - **Installation**: `pip install mcp-server-fetch`
   - **Repository**: https://github.com/modelcontextprotocol/servers/tree/main/src/fetch
   - **Tools**: `fetch` - Fetches URLs and extracts content

2. **🎨 Canva Dev MCP Server**
   - **Purpose**: AI-powered Canva development assistance
   - **Features**: Canva app development, API guidance, best practices
   - **Installation**: `npx -y @canva/cli@latest mcp` (requires Node.js)
   - **Documentation**: https://www.canva.dev/docs/apps/mcp-server/
   - **Tools**: Various Canva development assistance tools

### Why Use MCP?

```
Traditional Approach:
┌────────┐    Custom      ┌──────────┐
│  LLM   │◄─ Integration ─┤ Database │
└────────┘                └──────────┘
    │
    │      Custom
    │    Integration     ┌──────────┐
    └─────────────────►│   Files  │
                        └──────────┘

MCP Approach:
┌────────┐                ┌──────────┐
│  LLM   │◄── MCP ───────┤  Fetch   │
└────────┘                ├──────────┤
    │                     │  Canva   │
    │                     ├──────────┤
    └────── MCP ──────────┤  Custom  │
                          └──────────┘

✓ Standardized interface
✓ Reusable across projects
✓ Built-in security
✓ Real production servers
```

### Quick Start

#### 1. Installation

```bash
# Install Fetch MCP Server
pip install mcp-server-fetch

# Canva MCP Server requires Node.js and npm
# It will be installed automatically when first used
# Ensure you have Node.js v20+ and npm installed
```

#### 2. Basic MCP Setup with Fetch Server

```python
from src.mcp_integration import MCPClient, get_fetch_server_config
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

# Connect to Fetch MCP server
fetch_client = MCPClient(get_fetch_server_config())
fetch_client.start()

# Create tool schema
class FetchInput(BaseModel):
    url: str = Field(description="URL to fetch")
    max_length: int = Field(default=5000, description="Max characters")

# Create LangChain tool
def fetch_url(url: str, max_length: int = 5000) -> str:
    return fetch_client.call_tool("fetch", {
        "url": url,
        "max_length": max_length
    })

fetch_tool = StructuredTool.from_function(
    func=fetch_url,
    name="fetch_webpage",
    description="Fetch web content as markdown",
    args_schema=FetchInput
)

# Use in agent
agent = create_tool_calling_agent(llm, [fetch_tool], prompt)
```

#### 3. Multiple MCP Servers (Fetch + Canva)

```python
from src.mcp_integration import (
    MCPClient, 
    get_fetch_server_config,
    get_canva_server_config
)

# Connect to multiple servers
fetch_client = MCPClient(get_fetch_server_config())
canva_client = MCPClient(get_canva_server_config())

fetch_client.start()
canva_client.start()

# Create tools from both servers
tools = []

# Add Fetch tool
def fetch_url(url: str, max_length: int = 5000) -> str:
    return fetch_client.call_tool("fetch", {"url": url, "max_length": max_length})

tools.append(StructuredTool.from_function(
    func=fetch_url,
    name="fetch_webpage",
    description="Fetch and read web page content"
))

# Add Canva tools (example with first available tool)
if canva_client.tools:
    canva_tool = canva_client.tools[0]
    
    def canva_help(query: str) -> str:
        return canva_client.call_tool(canva_tool['name'], {"query": query})
    
    tools.append(StructuredTool.from_function(
        func=canva_help,
        name="canva_dev_help",
        description=canva_tool.get('description', 'Canva development assistance')
    ))
```

### MCP Architecture (with Real Servers)

```
┌─────────────────────────────────────────────┐
│         Your LangChain Application          │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │      Agent with MCP Tools            │  │
│  │  • fetch_webpage (Fetch Server)      │  │
│  │  • canva_dev_help (Canva Server)     │  │
│  │  • custom_tools                      │  │
│  └──────────────┬───────────────────────┘  │
│                 │                           │
└─────────────────┼───────────────────────────┘
                  │
                  │ MCP Protocol
                  │ (JSON-RPC over stdio)
                  │
┌─────────────────┴───────────────────────────┐
│           MCP Server Layer                  │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │  Fetch   │  │  Canva   │  │  Custom  │ │
│  │  Server  │  │  Server  │  │  Server  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
└───────┼─────────────┼─────────────┼────────┘
        │             │             │
┌───────▼─────┐ ┌────▼──────┐ ┌────▼─────────┐
│ Web Content │ │   Canva   │ │ Your Data    │
│   (HTTP)    │ │    API    │ │   Sources    │
└─────────────┘ └───────────┘ └──────────────┘

Communication Flow:
1. Agent decides to use MCP tool
2. Python code calls MCPClient.call_tool()
3. Request sent via JSON-RPC to MCP server process
4. Server executes tool (fetches web, queries Canva, etc.)
5. Response returned via JSON-RPC
6. Result passed back to agent
```

### MCP Resources

**Resources** provide read-only access to data:

```python
# List available resources
resources = mcp_server.list_resources()
# [
#   Resource(uri="file://docs/readme.md", name="readme"),
#   Resource(uri="db://users/table", name="users"),
# ]

# Read a specific resource
content = mcp_server.read_resource("file://docs/readme.md")
```

### MCP Tools

**Tools** allow LLMs to take actions:

```python
# Call a tool
result = mcp_server.call_tool(
    name="update_record",
    arguments={
        "table": "users",
        "id": 123,
        "data": {"status": "active"}
    }
)
```

### Security Best Practices

```python
class SecureMCPServer:
    """Secure MCP server implementation."""
    
    def __init__(self):
        self.allowed_uris = set()
        self.rate_limiter = RateLimiter()
    
    def read_resource(self, uri: str):
        # 1. Validate URI
        if not self.validate_uri(uri):
            raise SecurityError("Invalid URI")
        
        # 2. Check permissions
        if not self.check_permissions(uri):
            raise PermissionError("Access denied")
        
        # 3. Rate limiting
        if not self.rate_limiter.allow():
            raise RateLimitError("Too many requests")
        
        # 4. Sanitize and read
        return self.safe_read(uri)
```

### Production Example with Real Servers

```python
# production_mcp_agent.py

from src.mcp_integration import MCPClient, get_fetch_server_config, get_canva_server_config
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

class ProductionMCPAgent:
    def __init__(self):
        # Initialize real MCP servers
        self.servers = {}
        
        # Fetch Server
        try:
            fetch_config = get_fetch_server_config()
            fetch_client = MCPClient(fetch_config)
            if fetch_client.start():
                self.servers["fetch"] = fetch_client
                print("✅ Fetch server connected")
        except Exception as e:
            print(f"⚠️ Fetch server unavailable: {e}")
        
        # Canva Server
        try:
            canva_config = get_canva_server_config()
            canva_client = MCPClient(canva_config)
            if canva_client.start():
                self.servers["canva"] = canva_client
                print("✅ Canva server connected")
        except Exception as e:
            print(f"⚠️ Canva server unavailable: {e}")
        
        # Create tools
        self.tools = self.create_mcp_tools()
        
        # Setup monitoring
        self.setup_monitoring()
    
    def create_mcp_tools(self):
        """Create tools from real MCP servers."""
        tools = []
        
        # Fetch server tools
        if "fetch" in self.servers:
            fetch_client = self.servers["fetch"]
            
            class FetchInput(BaseModel):
                url: str = Field(description="URL to fetch")
                max_length: int = Field(default=5000, description="Max characters")
            
            def fetch_url(url: str, max_length: int = 5000) -> str:
                return self.wrap_tool_call(
                    "fetch", 
                    lambda: fetch_client.call_tool("fetch", {
                        "url": url, 
                        "max_length": max_length
                    })
                )
            
            tools.append(StructuredTool.from_function(
                func=fetch_url,
                name="fetch_webpage",
                description="Fetch web page content as markdown",
                args_schema=FetchInput
            ))
        
        # Canva server tools
        if "canva" in self.servers:
            canva_client = self.servers["canva"]
            
            for tool_def in canva_client.tools:
                # Create a tool for each Canva capability
                tools.append(self.create_canva_tool(tool_def, canva_client))
        
        return tools
    
    def wrap_tool_call(self, tool_name: str, func):
        """Wrap tool call with error handling and monitoring."""
        try:
            # Log call
            self.log_tool_call(tool_name)
            
            # Execute
            result = func()
            
            # Track success
            self.track_success(tool_name)
            
            return result
            
        except Exception as e:
            # Track error
            self.track_error(tool_name, e)
            raise
    
    def cleanup(self):
        """Cleanup MCP server connections."""
        for server in self.servers.values():
            server.stop()
```

---

## 📈 LangSmith Monitoring

### What is LangSmith?

**LangSmith** is LangChain's observability platform for:
- 🔍 Tracing LLM calls
- 📊 Performance monitoring
- 💰 Cost tracking
- 🐛 Debugging
- 📈 Analytics

### Setup

#### 1. Get API Key

1. Sign up at https://smith.langchain.com/
2. Create a project
3. Get your API key

#### 2. Configure Environment

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key_here
LANGCHAIN_PROJECT=my-project
```

#### 3. Verify Setup

```python
import os

print("Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
print("Project:", os.getenv("LANGCHAIN_PROJECT"))
```

### Automatic Tracing

Once configured, ALL LangChain operations are automatically traced:

```python
# This is automatically traced
llm = ChatOpenAI()
response = llm.invoke("Hello!")

# Agent execution is traced
agent_executor.invoke({"input": "Query"})

# Chains are traced
chain = prompt | llm | parser
result = chain.invoke({"topic": "AI"})
```

### Custom Callbacks

Add custom metrics:

```python
from langchain.callbacks import StdOutCallbackHandler
import time

class CustomMonitoring(StdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.metrics = {
            "llm_calls": 0,
            "total_tokens": 0,
            "errors": 0,
        }
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.metrics["llm_calls"] += 1
        self.start_time = time.time()
    
    def on_llm_end(self, response, **kwargs):
        duration = time.time() - self.start_time
        
        # Extract tokens
        if hasattr(response, 'llm_output'):
            tokens = response.llm_output.get('token_usage', {})
            self.metrics["total_tokens"] += tokens.get('total_tokens', 0)
        
        # Log to custom system
        self.log_metric("llm_duration", duration)
    
    def on_llm_error(self, error, **kwargs):
        self.metrics["errors"] += 1
        self.log_error(error)

# Use custom callback
callback = CustomMonitoring()
agent_executor.invoke(
    {"input": "Query"},
    callbacks=[callback]
)
```

### Cost Tracking

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    # Execute operations
    response1 = llm.invoke("Query 1")
    response2 = llm.invoke("Query 2")
    
    # Get costs
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Total Cost: ${cb.total_cost:.4f}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
```

### LangSmith Dashboard

Access your dashboard at: https://smith.langchain.com/

**Features:**
- 📊 **Traces**: See every LLM call and agent step
- 💰 **Cost Analytics**: Track spending over time
- ⚡ **Performance**: Latency and throughput metrics
- 🐛 **Debugging**: Replay failed requests
- 📈 **Trends**: Historical analysis

### Production Monitoring Setup

```python
# monitoring.py

import os
from langsmith import Client
from datetime import datetime

class ProductionMonitoring:
    def __init__(self):
        # Configure LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_KEY")
        os.environ["LANGCHAIN_PROJECT"] = "production"
        
        self.client = Client()
    
    def log_run(self, run_type, inputs, outputs, metadata=None):
        """Log a custom run."""
        return self.client.create_run(
            name=run_type,
            inputs=inputs,
            outputs=outputs,
            run_type="chain",
            extra=metadata or {}
        )
    
    def add_feedback(self, run_id, score, comment=None):
        """Add user feedback to a run."""
        self.client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=score,
            comment=comment
        )
    
    def create_dataset(self, name, examples):
        """Create a test dataset."""
        dataset = self.client.create_dataset(name)
        
        for example in examples:
            self.client.create_example(
                dataset_id=dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"]
            )
        
        return dataset
    
    def evaluate(self, dataset_name, agent):
        """Evaluate agent on dataset."""
        from langchain.smith import RunEvalConfig
        
        eval_config = RunEvalConfig(
            evaluators=["qa", "context_qa"],
            custom_evaluators=[]
        )
        
        results = self.client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=lambda: agent,
            evaluation=eval_config
        )
        
        return results
```

### Alerts Configuration

```python
# alerts.py

class MonitoringAlerts:
    def __init__(self):
        self.thresholds = {
            "error_rate": 0.05,  # 5%
            "latency_p95": 5.0,  # 5 seconds
            "cost_per_hour": 10.0,  # $10/hour
        }
    
    def check_error_rate(self, metrics):
        """Check if error rate exceeds threshold."""
        error_rate = metrics["errors"] / metrics["total_requests"]
        
        if error_rate > self.thresholds["error_rate"]:
            self.send_alert(
                level="warning",
                message=f"Error rate {error_rate:.2%} exceeds threshold"
            )
    
    def check_latency(self, metrics):
        """Check P95 latency."""
        p95_latency = self.calculate_p95(metrics["latencies"])
        
        if p95_latency > self.thresholds["latency_p95"]:
            self.send_alert(
                level="warning",
                message=f"P95 latency {p95_latency:.2f}s exceeds threshold"
            )
    
    def send_alert(self, level, message):
        """Send alert via email/Slack/etc."""
        print(f"[{level.upper()}] {message}")
        # Integrate with alerting service
```

---

## 🎯 Best Practices

### MCP Best Practices

1. **Security First**
   - Validate all URIs
   - Implement authentication
   - Use least-privilege access
   - Sanitize inputs

2. **Error Handling**
   - Graceful degradation
   - Retry logic
   - Clear error messages
   - Circuit breakers

3. **Performance**
   - Cache frequently accessed resources
   - Use connection pooling
   - Implement rate limiting
   - Monitor response times

### Monitoring Best Practices

1. **Comprehensive Tracing**
   - Trace all production traffic
   - Use sampling for high volume
   - Tag traces with metadata
   - Retain traces for analysis

2. **Cost Management**
   - Set budget alerts
   - Monitor token usage
   - Optimize prompts
   - Cache when possible

3. **Performance Optimization**
   - Track P50, P95, P99 latencies
   - Identify slow operations
   - Optimize bottlenecks
   - Use async where possible

---

## 🚀 Production Deployment

### Checklist

```
MCP Integration:
[ ] MCP servers configured
[ ] Authentication implemented
[ ] Rate limiting enabled
[ ] Error handling robust
[ ] Resources documented
[ ] Tools tested
[ ] Security review complete

Monitoring:
[ ] LangSmith configured
[ ] Tracing enabled
[ ] Cost tracking active
[ ] Alerts set up
[ ] Dashboard created
[ ] Team access granted
[ ] Retention policy set

Performance:
[ ] Baseline metrics established
[ ] SLOs defined
[ ] Load testing complete
[ ] Caching implemented
[ ] Connection pooling configured

Security:
[ ] API keys secured
[ ] PII filtering enabled
[ ] Access controls configured
[ ] Audit logging enabled
[ ] Compliance verified
```

### Example Production Configuration

```python
# config/production.py

class ProductionConfig:
    # MCP Configuration
    MCP_SERVERS = {
        "database": {
            "url": os.getenv("DB_MCP_URL"),
            "timeout": 30,
            "max_connections": 10,
            "retry_attempts": 3,
        },
        "filesystem": {
            "root_path": "/data",
            "read_only": True,
            "allowed_extensions": [".txt", ".json", ".csv"],
        },
    }
    
    # Monitoring Configuration
    LANGSMITH_CONFIG = {
        "tracing": True,
        "sampling_rate": 1.0,  # 100% in production
        "project": "production",
        "tags": ["prod", "v1.0"],
    }
    
    # Alert Thresholds
    ALERTS = {
        "error_rate_threshold": 0.05,
        "latency_p95_threshold": 5.0,
        "cost_hourly_threshold": 10.0,
    }
    
    # Performance
    CACHE_CONFIG = {
        "enabled": True,
        "ttl": 3600,
        "max_size": 1000,
    }
```

---

## 📚 Additional Resources

- **MCP Documentation**: https://modelcontextprotocol.io/
- **LangSmith Docs**: https://docs.smith.langchain.com/
- **LangChain Docs**: https://python.langchain.com/
- **Example Code**: See `src/07_mcp/` and `src/08_monitoring/`

---

**Ready to build production-ready AI systems!** 🚀

