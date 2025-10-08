# 🔌 MCP Servers Setup Guide

Quick reference for setting up and using the real MCP servers integrated in this project.

## 📋 Overview

This project uses **two real production MCP servers**:

1. **🌐 Fetch MCP Server** - Web content retrieval and processing
2. **🎨 Canva Dev MCP Server** - Canva development assistance

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+ (required for Fetch server)
python --version

# Node.js v20+ and npm (required for Canva server)
node --version
npm --version
```

### Installation

#### 1. Fetch MCP Server

```bash
# Install via pip
pip install mcp-server-fetch

# Or install with the project requirements
pip install -r requirements.txt
```

**Verification:**
```bash
# Test the server
python -m mcp_server_fetch
# Press Ctrl+C to stop
```

#### 2. Canva Dev MCP Server

```bash
# No pre-installation needed!
# The server is installed automatically when first used via npx

# Verify Node.js and npm are installed
node --version  # Should be v20 or higher
npm --version
```

**Manual installation (optional):**
```bash
npm install -g @canva/cli
```

## 🔧 Configuration

### Fetch Server Configuration

The Fetch server is configured in `src/07_mcp/mcp_integration.py`:

```python
def get_fetch_server_config() -> MCPServerConfig:
    return MCPServerConfig(
        name="fetch",
        command="python",
        args=["-m", "mcp_server_fetch"]
    )
```

### Canva Server Configuration

The Canva server is configured in `src/07_mcp/mcp_integration.py`:

```python
def get_canva_server_config() -> MCPServerConfig:
    return MCPServerConfig(
        name="canva-dev",
        command="npx",
        args=["-y", "@canva/cli@latest", "mcp"]
    )
```

## 📚 Usage Examples

### Basic Usage - Fetch Server

```python
from src.mcp_integration import MCPClient, get_fetch_server_config

# Initialize and start the server
fetch_client = MCPClient(get_fetch_server_config())
fetch_client.start()

# Use the fetch tool
result = fetch_client.call_tool("fetch", {
    "url": "https://example.com",
    "max_length": 5000
})

print(result)

# Cleanup
fetch_client.stop()
```

### Basic Usage - Canva Server

```python
from src.mcp_integration import MCPClient, get_canva_server_config

# Initialize and start the server
canva_client = MCPClient(get_canva_server_config())
canva_client.start()

# List available tools
for tool in canva_client.tools:
    print(f"Tool: {tool['name']}")
    print(f"Description: {tool.get('description', 'N/A')}")

# Cleanup
canva_client.stop()
```

### Using with LangChain Agents

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field

# Start MCP client
fetch_client = MCPClient(get_fetch_server_config())
fetch_client.start()

# Create tool schema
class FetchInput(BaseModel):
    url: str = Field(description="URL to fetch")
    max_length: int = Field(default=5000, description="Max characters")

# Create LangChain tool
def fetch_url(url: str, max_length: int = 5000) -> str:
    return fetch_client.call_tool("fetch", {"url": url, "max_length": max_length})

fetch_tool = StructuredTool.from_function(
    func=fetch_url,
    name="fetch_webpage",
    description="Fetch and convert web page content to markdown",
    args_schema=FetchInput
)

# Create agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant with web access."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, [fetch_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[fetch_tool], verbose=True)

# Use the agent
result = agent_executor.invoke({
    "input": "Fetch the content from https://example.com and summarize it"
})

print(result["output"])

# Cleanup
fetch_client.stop()
```

## 🎯 Available Tools

### Fetch Server Tools

#### `fetch`
Fetches a URL from the internet and extracts its contents as markdown.

**Parameters:**
- `url` (string, required): URL to fetch
- `max_length` (integer, optional): Maximum number of characters to return (default: 5000)
- `start_index` (integer, optional): Start content from this character index (default: 0)
- `raw` (boolean, optional): Get raw content without markdown conversion (default: false)

**Example:**
```python
result = fetch_client.call_tool("fetch", {
    "url": "https://www.example.com",
    "max_length": 10000,
    "raw": False
})
```

### Canva Server Tools

The Canva Dev MCP server provides multiple tools for Canva development assistance. To see all available tools:

```python
canva_client = MCPClient(get_canva_server_config())
canva_client.start()

for tool in canva_client.tools:
    print(f"\nTool: {tool['name']}")
    print(f"Description: {tool.get('description', 'N/A')}")
    
    if 'inputSchema' in tool:
        schema = tool['inputSchema']
        if 'properties' in schema:
            print("Parameters:")
            for param_name, param_info in schema['properties'].items():
                print(f"  - {param_name}: {param_info.get('description', 'N/A')}")
```

## 🏃 Running the Examples

### Run all MCP examples

```bash
cd langchain-agent-demo
python src/07_mcp/mcp_integration.py
```

### Run specific examples

The examples are interactive and will guide you through:
1. **MCP Basics** - Understanding and connecting to servers
2. **Fetch Tool in Agent** - Using web fetching in agents
3. **Multiple Servers** - Combining Fetch and Canva
4. **Best Practices** - Production patterns
5. **Real-World Scenario** - Complete research assistant example

## ⚠️ Troubleshooting

### Fetch Server Issues

**Problem:** `ModuleNotFoundError: No module named 'mcp_server_fetch'`

**Solution:**
```bash
pip install mcp-server-fetch
```

**Problem:** Server exits immediately

**Solution:**
- Ensure Python 3.11+ is installed
- Check for error messages in the output
- Try running manually: `python -m mcp_server_fetch`

### Canva Server Issues

**Problem:** `command not found: npx`

**Solution:**
```bash
# Install Node.js and npm
# Visit: https://nodejs.org/

# Verify installation
node --version
npm --version
```

**Problem:** Server takes long to start

**Solution:**
- First run downloads @canva/cli (normal behavior)
- Subsequent runs will be faster
- Use `-y` flag to skip prompts (already included in config)

### General MCP Issues

**Problem:** Connection timeout or no response

**Solution:**
1. Check server logs for errors
2. Ensure the server process is running
3. Verify JSON-RPC communication is working
4. Try restarting the server

**Problem:** Tool calls return errors

**Solution:**
1. Verify tool parameters match the schema
2. Check input validation
3. Review server logs for details
4. Test tool independently before using in agent

## 🔒 Security Considerations

### Fetch Server

⚠️ **Important:** The Fetch server can access local and internal IP addresses, which may pose security risks.

**Best Practices:**
- Implement URL allowlists in production
- Set reasonable `max_length` limits
- Monitor fetch requests
- Consider caching to reduce external calls
- Validate URLs before fetching

**Example with validation:**
```python
def safe_fetch(url: str, max_length: int = 5000) -> str:
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise ValueError("Only HTTP/HTTPS URLs allowed")
    
    # Prevent local access
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
        raise ValueError("Cannot access local resources")
    
    # Fetch safely
    return fetch_client.call_tool("fetch", {"url": url, "max_length": max_length})
```

### Canva Server

- May require API keys for full functionality
- Review Canva's security documentation
- Keep @canva/cli updated

## 📊 Monitoring

All MCP operations can be monitored with LangSmith:

```python
import os

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key"
os.environ["LANGCHAIN_PROJECT"] = "mcp-integration"

# Now all MCP tool calls will be traced
```

View traces at: https://smith.langchain.com/

## 📖 Additional Resources

### Official Documentation

- **MCP Specification**: https://modelcontextprotocol.io/
- **Fetch Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/fetch
- **Canva Dev MCP**: https://www.canva.dev/docs/apps/mcp-server/
- **LangChain**: https://python.langchain.com/

### Project Documentation

- `MCP_MONITORING_GUIDE.md` - Comprehensive MCP integration guide
- `EXAMPLES_GUIDE.md` - All example documentation
- `QUICKSTART.md` - Quick project setup
- `ARCHITECTURE.md` - System architecture

## 🤝 Contributing

Found an issue or want to add a new MCP server? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Add your MCP server configuration
4. Update documentation
5. Submit a pull request

## 📝 License

This project is provided as educational material. See main project README for license details.

---

**Questions?** Check the main documentation or open an issue on GitHub.

**Happy coding with MCP!** 🚀

