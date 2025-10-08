"""
MCP (Model Context Protocol) Integration with Real Servers
===========================================================

This module demonstrates integration with real MCP servers:
- Fetch MCP Server (web content fetching)
- Canva Dev MCP Server (Canva development assistance)

Learning Objectives:
- Understand Model Context Protocol
- Connect to real MCP servers
- Use MCP tools in agents
- Access MCP resources
- Integrate with LangChain workflows
"""

import sys
from pathlib import Path
import json
import subprocess
import asyncio
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from src.utils.display import print_section, print_response, print_step, print_error
from config.settings import settings, validate_api_keys


# ============================================================================
# MCP Concept Explanation
# ============================================================================

def explain_mcp():
    """Explain what MCP is and why it's useful."""
    explanation = """
    🔌 Model Context Protocol (MCP)
    
    MCP is an open protocol that standardizes how applications provide
    context to Large Language Models (LLMs). Think of it as a universal
    adapter that lets AI assistants connect to any data source, tool,
    or service through a standard interface.
    
    Key Concepts:
    
    1. 📦 RESOURCES
       - Expose data from your application to LLMs
       - Can be files, database records, API responses, etc.
       - Read-only access to context
       
    2. 🛠️  TOOLS
       - Allow LLMs to take actions in your application
       - Can create, update, or delete data
       - Extend agent capabilities
       
    3. 💬 PROMPTS
       - Reusable prompt templates
       - Can include context from resources
       - Standardized across applications
    
    4. 🔌 SERVERS
       - MCP servers expose capabilities
       - One server can provide multiple resources/tools
       - Runs as separate process or service
       - Communicates via JSON-RPC over stdio
    
    Benefits:
    - 🔄 Standardization: One protocol for all integrations
    - 🚀 Extensibility: Easy to add new capabilities
    - 🔒 Security: Controlled access to resources
    - 🎯 Modularity: Mix and match different servers
    
    Real Servers Used in This Demo:
    - 🌐 Fetch Server: Retrieve and process web content
    - 🎨 Canva Server: AI-powered Canva development assistance
    """
    return explanation


# ============================================================================
# MCP Client Implementation
# ============================================================================

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None


class MCPClient:
    """
    Client for communicating with MCP servers via JSON-RPC over stdio.
    
    This implements the Model Context Protocol for connecting to
    external MCP servers like Fetch and Canva.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.tools = []
        self.resources = []
        self.prompts = []
        self._connected = False
    
    def start(self) -> bool:
        """Start the MCP server process."""
        try:
            print(f"   Starting MCP server: {self.config.name}")
            print(f"   Command: {self.config.command} {' '.join(self.config.args)}")
            
            # Start the server process
            self.process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self.config.env
            )
            
            # Wait a moment for the server to start
            time.sleep(2)
            
            # Check if process is still running
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                raise Exception(f"Server process exited immediately. Error: {stderr}")
            
            # Initialize the connection
            if self._initialize():
                self._connected = True
                print(f"   ✅ Connected to {self.config.name}")
                return True
            else:
                print(f"   ❌ Failed to initialize {self.config.name}")
                return False
                
        except FileNotFoundError:
            print(f"   ❌ Command not found: {self.config.command}")
            print(f"   💡 Make sure to install the server:")
            if self.config.command == "python":
                print(f"      pip install mcp-server-fetch")
            elif self.config.command == "npx":
                print(f"      Ensure Node.js and npm are installed")
            return False
        except Exception as e:
            print(f"   ❌ Error starting server: {e}")
            if self.process and self.process.stderr:
                stderr = self.process.stderr.read()
                if stderr:
                    print(f"   Server error: {stderr}")
            return False
    
    def _initialize(self) -> bool:
        """Initialize the MCP connection."""
        try:
            # Send initialize request
            response = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": False},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "langchain-mcp-client",
                    "version": "1.0.0"
                }
            })
            
            if response and "result" in response:
                # Send initialized notification
                self._send_notification("notifications/initialized", {})
                
                # List available tools
                self._list_tools()
                
                return True
            return False
            
        except Exception as e:
            print(f"   Error during initialization: {e}")
            return False
    
    def _send_request(self, method: str, params: Dict) -> Optional[Dict]:
        """Send a JSON-RPC request to the server."""
        if not self.process or not self.process.stdin:
            return None
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()
            
            # Read response (with timeout)
            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line)
            return None
            
        except Exception as e:
            print(f"   Error in request/response: {e}")
            return None
    
    def _send_notification(self, method: str, params: Dict):
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process or not self.process.stdin:
            return
        
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        
        try:
            notification_line = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_line)
            self.process.stdin.flush()
        except Exception as e:
            print(f"   Error sending notification: {e}")
    
    def _list_tools(self) -> List[Dict]:
        """List available tools from the server."""
        response = self._send_request("tools/list", {})
        if response and "result" in response and "tools" in response["result"]:
            self.tools = response["result"]["tools"]
            print(f"   📦 Found {len(self.tools)} tools")
            return self.tools
        return []
    
    def call_tool(self, tool_name: str, arguments: Dict) -> str:
        """Call a tool on the MCP server."""
        if not self._connected:
            return f"Error: Not connected to {self.config.name}"
        
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if response and "result" in response:
            result = response["result"]
            
            # Extract content from the response
            if "content" in result:
                content_items = result["content"]
                if isinstance(content_items, list) and len(content_items) > 0:
                    first_item = content_items[0]
                    if "text" in first_item:
                        return first_item["text"]
                    return str(first_item)
                return str(content_items)
            
            return str(result)
        
        if response and "error" in response:
            return f"Error: {response['error'].get('message', 'Unknown error')}"
        
        return "No response from server"
    
    def list_resources(self) -> List[Dict]:
        """List available resources from the server."""
        response = self._send_request("resources/list", {})
        if response and "result" in response and "resources" in response["result"]:
            self.resources = response["result"]["resources"]
            return self.resources
        return []
    
    def stop(self):
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self._connected = False


# ============================================================================
# MCP Server Configurations
# ============================================================================

def get_fetch_server_config() -> MCPServerConfig:
    """Get configuration for Fetch MCP server."""
    return MCPServerConfig(
        name="fetch",
        command="python",
        args=["-m", "mcp_server_fetch"]
    )


def get_canva_server_config() -> MCPServerConfig:
    """Get configuration for Canva Dev MCP server."""
    return MCPServerConfig(
        name="canva-dev",
        command="npx",
        args=["-y", "@canva/cli@latest", "mcp"]
    )


# ============================================================================
# Examples
# ============================================================================

def example_1_mcp_basics():
    """
    Example 1: MCP Basics
    Understanding MCP concepts and connecting to real servers
    """
    print_section(
        "Example 1: MCP Basics",
        "Understanding Model Context Protocol with Real Servers"
    )
    
    print_step(1, "What is MCP?")
    print(explain_mcp())
    
    print_step(2, "Connect to Fetch MCP Server")
    
    fetch_client = MCPClient(get_fetch_server_config())
    fetch_started = fetch_client.start()
    
    if fetch_started:
        print(f"\n   📦 Available tools in Fetch server:")
        for tool in fetch_client.tools:
            print(f"\n   Tool: {tool['name']}")
            print(f"   Description: {tool.get('description', 'N/A')}")
            if 'inputSchema' in tool:
                schema = tool['inputSchema']
                if 'properties' in schema:
                    print(f"   Parameters:")
                    for param_name, param_info in schema['properties'].items():
                        required = param_name in schema.get('required', [])
                        req_str = "required" if required else "optional"
                        print(f"     - {param_name} ({req_str}): {param_info.get('description', 'N/A')}")
        
        fetch_client.stop()
    
    print("\n" + "="*70)
    print_step(3, "Connect to Canva Dev MCP Server")
    
    print("   ℹ️  Note: Canva MCP server requires Node.js and npm")
    print("   If not installed, this connection will be skipped\n")
    
    canva_client = MCPClient(get_canva_server_config())
    canva_started = canva_client.start()
    
    if canva_started:
        print(f"\n   📦 Available tools in Canva Dev server:")
        for i, tool in enumerate(canva_client.tools[:5], 1):  # Show first 5
            print(f"   {i}. {tool['name']}: {tool.get('description', 'N/A')[:80]}")
        
        if len(canva_client.tools) > 5:
            print(f"   ... and {len(canva_client.tools) - 5} more tools")
        
        canva_client.stop()


def example_2_fetch_tool_in_agent():
    """
    Example 2: Using Fetch MCP Tool in LangChain Agent
    Integrate real MCP fetch tool with agents
    """
    print_section(
        "Example 2: Fetch MCP Tool in Agent",
        "Use real web fetching capabilities in agents"
    )
    
    print_step(1, "Initialize Fetch MCP Server")
    
    fetch_client = MCPClient(get_fetch_server_config())
    if not fetch_client.start():
        print("   ❌ Could not start Fetch server. Skipping this example.")
        print("   💡 Install with: pip install mcp-server-fetch")
        return
    
    print_step(2, "Create LangChain tool from MCP fetch")
    
    class FetchInput(BaseModel):
        """Input for fetch tool."""
        url: str = Field(description="URL to fetch content from")
        max_length: int = Field(default=5000, description="Maximum characters to return")
    
    def fetch_url(url: str, max_length: int = 5000) -> str:
        """Fetch content from a URL using MCP."""
        return fetch_client.call_tool("fetch", {
            "url": url,
            "max_length": max_length
        })
    
    fetch_tool = StructuredTool.from_function(
        func=fetch_url,
        name="fetch_webpage",
        description="Fetch and convert web page content to markdown. Input: URL and optional max_length",
        args_schema=FetchInput
    )
    
    print("   ✅ Fetch tool created")
    
    print_step(3, "Create agent with fetch capability")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful research assistant with the ability to fetch content from the web. "
         "Use the fetch_webpage tool to retrieve information from URLs when needed."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, [fetch_tool], prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[fetch_tool],
        verbose=True,
        handle_parsing_errors=True
    )
    
    print_step(4, "Test with real web queries")
    
    queries = [
        "Fetch the content from https://projectpath.ai and summarize what you find",
    ]
    
    for query in queries:
        print(f"\n   ❓ Query: {query}")
        try:
            result = agent_executor.invoke({"input": query})
            print_response(result["output"], "Agent Response")
        except Exception as e:
            print_error(f"Error: {e}")
        
        input("      Press Enter to continue...")
    
    # Cleanup
    fetch_client.stop()


def example_3_multiple_mcp_servers():
    """
    Example 3: Using Multiple MCP Servers Together
    Combine Fetch and Canva servers
    """
    print_section(
        "Example 3: Multiple MCP Servers",
        "Combine tools from Fetch and Canva servers"
    )
    
    print_step(1, "Initialize both MCP servers")
    
    fetch_client = MCPClient(get_fetch_server_config())
    canva_client = MCPClient(get_canva_server_config())
    
    fetch_started = fetch_client.start()
    canva_started = canva_client.start()
    
    if not fetch_started:
        print("   ⚠️  Fetch server not available")
    if not canva_started:
        print("   ⚠️  Canva server not available")
    
    if not (fetch_started or canva_started):
        print("   ❌ No servers available. Skipping this example.")
        return
    
    print_step(2, "Create unified tool set")
    
    tools = []
    
    # Add Fetch tool if available
    if fetch_started:
        class FetchInput(BaseModel):
            url: str = Field(description="URL to fetch")
            max_length: int = Field(default=5000, description="Max characters")
        
        def fetch_url(url: str, max_length: int = 5000) -> str:
            return fetch_client.call_tool("fetch", {"url": url, "max_length": max_length})
        
        tools.append(StructuredTool.from_function(
            func=fetch_url,
            name="fetch_webpage",
            description="Fetch web page content as markdown",
            args_schema=FetchInput
        ))
        print("   ✅ Added Fetch tools")
    
    # Add Canva tools if available
    if canva_started:
        # Create a simple generic input model for Canva tools (outside loop)
        # Using Optional fields to handle various tool signatures
        class CanvaToolInput(BaseModel):
            """Generic input for Canva tools"""
            query: Optional[str] = Field(None, description="Query or topic for the Canva tool")
            pages: Optional[List[str]] = Field(None, description="List of page slugs to read")
            component_name: Optional[str] = Field(None, description="Component name to look up")
        
        # Add multiple Canva tools to showcase its capabilities
        for canva_tool in canva_client.tools[:3]:  # Add first 3 tools as examples
            tool_name = canva_tool['name']
            tool_description = canva_tool.get('description', 'Canva development tool')
            
            # Get input schema from the tool
            input_schema = canva_tool.get('inputSchema', {})
            properties = input_schema.get('properties', {})
            
            def make_canva_func(tool_name_capture, properties_capture):
                def canva_func(**kwargs) -> str:
                    # Filter out None values and only pass relevant parameters
                    params = {k: v for k, v in kwargs.items() if v is not None}
                    
                    # Match kwargs to actual tool parameters
                    tool_params = {}
                    for prop_name in properties_capture.keys():
                        if prop_name in params:
                            tool_params[prop_name] = params[prop_name]
                    
                    # If no matching params but query provided, use it as the first parameter
                    if not tool_params and 'query' in params:
                        first_prop = list(properties_capture.keys())[0] if properties_capture else 'query'
                        tool_params[first_prop] = params['query']
                    
                    return canva_client.call_tool(tool_name_capture, tool_params)
                return canva_func
            
            tools.append(StructuredTool.from_function(
                func=make_canva_func(tool_name, properties),
                name=tool_name.replace('-', '_'),  # Make name Python-compatible
                description=tool_description,
                args_schema=CanvaToolInput
            ))
        
        print("   ✅ Added Canva tools")
    
    print(f"   📦 Total tools available: {len(tools)}")
    
    print_step(3, "Create multi-server agent")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant with access to multiple capabilities:\n"
         "- Web content fetching (via Fetch MCP)\n"
         "- Canva development assistance (via Canva MCP)\n"
         "Use the appropriate tools based on the user's request."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print_step(4, "Test with multi-tool queries")
    
    # First query: Use Fetch tool
    if fetch_started:
        query = "Fetch content from https://www.projectpath.ai and tell me what it's about"
        print(f"\n   ❓ Query 1 (Fetch): {query}")
        
        try:
            result = agent_executor.invoke({"input": query})
            print_response(result["output"], "Agent Response")
        except Exception as e:
            print_error(f"Error: {e}")
    
    # Second query: Use Canva tool
    if canva_started:
        print("\n")
        canva_query = "I want to create a Canva app that helps users design project roadmaps. How do I get started with Canva app development?"
        print(f"   ❓ Query 2 (Canva): {canva_query}")
        
        try:
            result = agent_executor.invoke({"input": canva_query})
            print_response(result["output"], "Agent Response")
        except Exception as e:
            print_error(f"Error: {e}")
    
    # Cleanup
    if fetch_started:
        fetch_client.stop()
    if canva_started:
        canva_client.stop()


def example_4_mcp_best_practices():
    """
    Example 4: MCP Best Practices
    Guidelines for production MCP integration
    """
    print_section(
        "Example 4: MCP Best Practices",
        "Production-ready MCP integration patterns"
    )
    
    practices = """
    🎯 Best Practices for Real MCP Integration
    
    1. 🔒 SECURITY
       ✓ Validate all URLs and inputs
       ✓ Be cautious with fetch server (can access local IPs)
       ✓ Implement rate limiting
       ✓ Use authentication where required
       ✓ Log all MCP operations
    
    2. 📊 MONITORING
       ✓ Track MCP server health
       ✓ Monitor tool usage and errors
       ✓ Set up alerts for server failures
       ✓ Use LangSmith for agent tracing
       ✓ Log all tool calls and results
    
    3. 🚀 PERFORMANCE
       ✓ Cache frequently fetched content
       ✓ Set appropriate timeouts
       ✓ Handle slow responses gracefully
       ✓ Consider async operations for multiple tools
       ✓ Restart servers if they become unresponsive
    
    4. 🔄 ERROR HANDLING
       ✓ Gracefully handle server startup failures
       ✓ Implement retry logic for tool calls
       ✓ Provide clear error messages to users
       ✓ Have fallback strategies
       ✓ Auto-restart failed servers
    
    5. 📝 DOCUMENTATION
       ✓ Document required dependencies (Node.js, pip packages)
       ✓ Provide clear setup instructions
       ✓ Document each tool's capabilities
       ✓ Include example queries
       ✓ Version your MCP configurations
    
    6. 🧪 TESTING
       ✓ Test server startup/shutdown
       ✓ Test each tool independently
       ✓ Integration test with agents
       ✓ Handle timeout scenarios
       ✓ Test with malformed inputs
    
    7. 🏗️  ARCHITECTURE
       ✓ Use connection pooling for multiple agents
       ✓ Implement health checks
       ✓ Graceful shutdown procedures
       ✓ Process isolation for stability
       ✓ Consider containerization
    
    8. 🌐 REAL SERVER SPECIFICS
       
       Fetch Server:
       ✓ Be aware it can access local/internal IPs
       ✓ Set max_length to prevent excessive data
       ✓ Consider caching fetched content
       ✓ Implement URL allowlists for security
       
       Canva Server:
       ✓ Requires Node.js and npm
       ✓ May need API keys for full functionality
       ✓ Keep @canva/cli updated
       ✓ Test with actual Canva projects
    """
    
    print(practices)
    
    print("\n" + "="*70)
    print("\n   📋 Production Checklist:\n")
    
    checklist = [
        "Dependencies installed (pip, npm, Node.js)",
        "MCP servers tested and working",
        "Error handling implemented",
        "Monitoring and logging set up",
        "Rate limiting configured",
        "Security review complete",
        "Documentation written",
        "Tests passing",
        "Graceful shutdown implemented",
    ]
    
    for i, item in enumerate(checklist, 1):
        print(f"   [ ] {i}. {item}")
    
    print()


def example_5_real_world_scenario():
    """
    Example 5: Real-World Scenario
    Research assistant with web access
    """
    print_section(
        "Example 5: Real-World Scenario",
        "Research assistant with MCP-powered web access"
    )
    
    print_step(1, "Scenario: AI Research Assistant")
    
    scenario = """
    Scenario: Research Assistant with Web Access
    
    The agent needs to:
    - 🌐 Fetch content from web pages
    - 📊 Summarize information
    - 🔍 Compare multiple sources
    - 📝 Generate reports
    
    Powered by:
    - Fetch MCP Server (web content retrieval)
    - GPT-4 (analysis and generation)
    """
    
    print(scenario)
    
    print_step(2, "Initialize Fetch MCP Server")
    
    fetch_client = MCPClient(get_fetch_server_config())
    if not fetch_client.start():
        print("   ❌ Could not start Fetch server. Skipping this example.")
        return
    
    print_step(3, "Create research tools")
    
    class FetchInput(BaseModel):
        url: str = Field(description="URL to fetch")
        max_length: int = Field(default=5000, description="Max characters")
    
    def fetch_url(url: str, max_length: int = 5000) -> str:
        """Fetch web content."""
        return fetch_client.call_tool("fetch", {"url": url, "max_length": max_length})
    
    tools = [
        StructuredTool.from_function(
            func=fetch_url,
            name="fetch_webpage",
            description="Fetch and read web page content",
            args_schema=FetchInput
        )
    ]
    
    print(f"   Created {len(tools)} research tools")
    
    print_step(4, "Create research agent")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert research assistant with web access. "
         "When given URLs, fetch their content and provide insightful analysis. "
         "Summarize key points, extract important information, and answer questions "
         "based on the fetched content."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    print_step(5, "Conduct research")
    
    request = """
    Please fetch the content from https://example.com and tell me:
    1. What is the main purpose of this page?
    2. What key information does it contain?
    3. Is there any contact information?
    """
    
    print(f"\n   🔬 Research Request:\n{request}\n")
    
    try:
        result = agent_executor.invoke({"input": request})
        print_response(result["output"], "Research Results")
    except Exception as e:
        print_error(f"Error: {e}")
    
    # Cleanup
    fetch_client.stop()


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" MCP (MODEL CONTEXT PROTOCOL) - REAL SERVER INTEGRATION".center(70))
    print("="*70 + "\n")
    
    print("   📌 This demo uses REAL MCP servers:")
    print("   - Fetch Server: Web content retrieval")
    print("   - Canva Dev Server: Canva development assistance")
    print("\n   ⚠️  Requirements:")
    print("   - Fetch: pip install mcp-server-fetch")
    print("   - Canva: Node.js, npm, @canva/cli")
    print("="*70 + "\n")
    
    try:
        example_1_mcp_basics()
        input("\nPress Enter to continue to next example...")
        
        example_2_fetch_tool_in_agent()
        input("\nPress Enter to continue to next example...")
        
        example_3_multiple_mcp_servers()
        input("\nPress Enter to continue to next example...")
        
        example_4_mcp_best_practices()
        input("\nPress Enter to continue to next example...")
        
        example_5_real_world_scenario()
        
        print("\n" + "="*70)
        print(" ✅ All MCP examples completed!".center(70))
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
