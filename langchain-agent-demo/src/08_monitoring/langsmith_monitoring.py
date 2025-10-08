"""
LangSmith Monitoring & Observability
====================================

This module demonstrates monitoring and debugging with LangSmith:
- Enable tracing
- Track agent execution
- Monitor token usage
- Debug failures
- Performance analysis
- Production monitoring

Learning Objectives:
- Set up LangSmith tracing
- Monitor agent behavior
- Analyze performance metrics
- Debug production issues
- Optimize costs
"""

import sys
import os
from pathlib import Path
from typing import Any
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_core.tracers import ConsoleCallbackHandler
from src.utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


# ============================================================================
# LangSmith Concepts
# ============================================================================

def explain_langsmith():
    """Explain LangSmith and its benefits."""
    explanation = """
    🔍 LangSmith - LLM Observability Platform
    
    LangSmith is a platform for monitoring, debugging, and improving
    LLM applications. It provides deep insights into your AI systems.
    
    Key Features:
    
    1. 📊 TRACING
       ├─ Track every LLM call
       ├─ See agent reasoning steps
       ├─ View tool executions
       └─ Measure latencies
    
    2. 📈 MONITORING
       ├─ Token usage analytics
       ├─ Cost tracking
       ├─ Performance metrics
       ├─ Error rates
       └─ Request volumes
    
    3. 🐛 DEBUGGING
       ├─ Replay failed requests
       ├─ Compare outputs
       ├─ Inspect prompts
       └─ Analyze errors
    
    4. ⚡ OPTIMIZATION
       ├─ Identify bottlenecks
       ├─ Reduce token usage
       ├─ Improve prompts
       └─ A/B testing
    
    5. 📝 DATASETS
       ├─ Create test cases
       ├─ Evaluate performance
       ├─ Regression testing
       └─ Version tracking
    
    6. 🎯 FEEDBACK
       ├─ Collect user ratings
       ├─ Track quality metrics
       ├─ Continuous improvement
       └─ Production insights
    
    Setup:
    1. Sign up at https://smith.langchain.com/
    2. Get your API key
    3. Set environment variables:
       LANGCHAIN_TRACING_V2=true
       LANGCHAIN_API_KEY=your_key
       LANGCHAIN_PROJECT=your_project
    
    Benefits:
    ✓ Understand agent behavior
    ✓ Optimize costs
    ✓ Debug production issues
    ✓ Improve quality
    ✓ Monitor performance
    """
    return explanation


# ============================================================================
# Custom Callback Handler for Demo
# ============================================================================

class DetailedMonitoringCallback(StdOutCallbackHandler):
    """Custom callback for detailed monitoring."""
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.llm_calls = 0
        self.tool_calls = 0
        self.errors = 0
        self.total_tokens = 0
    
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        """Track LLM call start."""
        self.llm_calls += 1
        self.start_time = time.time()
        print(f"\n   🤖 LLM Call #{self.llm_calls} started")
    
    def on_llm_end(self, response: Any, **kwargs):
        """Track LLM call end."""
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"   ✅ LLM Call completed in {duration:.2f}s")
            
            # Track tokens if available
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                total = token_usage.get('total_tokens', 0)
                self.total_tokens += total
                if total > 0:
                    print(f"   📊 Tokens used: {total} (Total: {self.total_tokens})")
    
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        """Track tool call start."""
        self.tool_calls += 1
        tool_name = serialized.get('name', 'unknown')
        print(f"\n   🔧 Tool Call #{self.tool_calls}: {tool_name}")
        print(f"      Input: {input_str[:100]}...")
    
    def on_tool_end(self, output: str, **kwargs):
        """Track tool call end."""
        print(f"   ✅ Tool completed")
        print(f"      Output: {output[:100]}...")
    
    def on_tool_error(self, error: Exception, **kwargs):
        """Track tool errors."""
        self.errors += 1
        print(f"   ❌ Tool Error #{self.errors}: {error}")
    
    def on_agent_action(self, action: Any, **kwargs):
        """Track agent actions."""
        print(f"\n   🤔 Agent Decision:")
        print(f"      Action: {action.tool}")
        print(f"      Reasoning: {action.log[:150]}...")
    
    def print_summary(self):
        """Print monitoring summary."""
        print("\n" + "="*70)
        print("   📊 MONITORING SUMMARY")
        print("="*70)
        print(f"   LLM Calls:        {self.llm_calls}")
        print(f"   Tool Calls:       {self.tool_calls}")
        print(f"   Errors:           {self.errors}")
        print(f"   Total Tokens:     {self.total_tokens}")
        print("="*70 + "\n")


# ============================================================================
# Examples
# ============================================================================

def example_1_enable_tracing():
    """
    Example 1: Enable LangSmith Tracing
    Basic tracing setup
    """
    print_section(
        "Example 1: Enable LangSmith Tracing",
        "Set up tracing for your LangChain applications"
    )
    
    print_step(1, "What is LangSmith?")
    print(explain_langsmith())
    
    print_step(2, "Check current tracing status")
    
    tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    project = os.environ.get("LANGCHAIN_PROJECT", "default")
    
    print(f"\n   Tracing Enabled: {tracing_enabled}")
    print(f"   API Key Set: {bool(api_key)}")
    print(f"   Project: {project}")
    
    if not tracing_enabled:
        print("\n   ⚠️  LangSmith tracing is not enabled")
        print("   To enable, set these environment variables in .env:")
        print("   LANGCHAIN_TRACING_V2=true")
        print("   LANGCHAIN_API_KEY=your_api_key")
        print("   LANGCHAIN_PROJECT=my-project")
    else:
        print("\n   ✅ LangSmith tracing is enabled!")
        print(f"   View traces at: https://smith.langchain.com/")
    
    print_step(3, "Example traced chain")
    
    # Simple chain that will be traced
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Tell me a fact about {topic}")
    chain = prompt | llm
    
    print("\n   Running a simple chain (will be traced if enabled)...")
    result = chain.invoke({"topic": "artificial intelligence"})
    
    print_response(result.content, "Chain Output")
    
    if tracing_enabled:
        print("\n   ✅ This execution was traced in LangSmith!")
        print("   Check your LangSmith dashboard to see the trace.")
    else:
        print("\n   💡 Enable tracing to see this in LangSmith dashboard")


def example_2_monitor_agent():
    """
    Example 2: Monitor Agent Execution
    Track agent behavior with custom callbacks
    """
    print_section(
        "Example 2: Monitor Agent Execution",
        "Track every step of agent execution"
    )
    
    print_step(1, "Create tools for agent")
    
    # Define input schemas
    class CalculatorInput(BaseModel):
        """Input for calculator tool."""
        expression: str = Field(description="Mathematical expression to evaluate")
    
    class TimeInput(BaseModel):
        """Input for time tool."""
        query: str = Field(default="", description="Query parameter (not used)")
    
    def calculator(expression: str) -> str:
        """Calculate mathematical expression."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    def get_time(query: str = "") -> str:
        """Get current time."""
        from datetime import datetime
        return f"Current time: {datetime.now().strftime('%H:%M:%S')}"
    
    tools = [
        StructuredTool.from_function(
            func=calculator,
            name="calculator",
            description="Calculate math expressions. Input should be a valid math expression.",
            args_schema=CalculatorInput
        ),
        StructuredTool.from_function(
            func=get_time,
            name="get_time",
            description="Get current time",
            args_schema=TimeInput
        ),
    ]
    
    print_step(2, "Create monitored agent")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use tools when needed."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create custom callback
    monitoring_callback = DetailedMonitoringCallback()
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Use our callback instead
        callbacks=[monitoring_callback],
        handle_parsing_errors=True
    )
    
    print_step(3, "Execute agent with monitoring")
    
    query = "What time is it? Then calculate 15 * 23."
    print(f"\n   ❓ Query: {query}\n")
    
    result = agent_executor.invoke({"input": query})
    
    print_response(result["output"], "Agent Response")
    
    print_step(4, "View monitoring summary")
    monitoring_callback.print_summary()


def example_3_track_costs():
    """
    Example 3: Track Costs and Token Usage
    Monitor API costs in real-time
    """
    print_section(
        "Example 3: Track Costs",
        "Monitor token usage and estimate costs"
    )
    
    print_step(1, "Create a chain with token tracking")
    
    from langchain.callbacks import get_openai_callback
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    
    prompts = [
        "Explain machine learning in one sentence.",
        "What is deep learning?",
        "How does neural network work?",
    ]
    
    print_step(2, "Execute with cost tracking")
    
    total_cost = 0
    total_tokens = 0
    
    for i, prompt_text in enumerate(prompts, 1):
        print(f"\n   📝 Request {i}: {prompt_text}")
        
        with get_openai_callback() as cb:
            response = llm.invoke(prompt_text)
            
            print(f"   💰 Tokens: {cb.total_tokens}")
            print(f"   💵 Cost: ${cb.total_cost:.6f}")
            print(f"   📊 Prompt: {cb.prompt_tokens} | Completion: {cb.completion_tokens}")
            
            total_cost += cb.total_cost
            total_tokens += cb.total_tokens
    
    print("\n" + "="*70)
    print("   💰 COST SUMMARY")
    print("="*70)
    print(f"   Total Requests:   {len(prompts)}")
    print(f"   Total Tokens:     {total_tokens}")
    print(f"   Total Cost:       ${total_cost:.6f}")
    print(f"   Avg Cost/Request: ${total_cost/len(prompts):.6f}")
    print("="*70 + "\n")


def example_4_debug_failures():
    """
    Example 4: Debug Failures
    Handle and debug errors
    """
    print_section(
        "Example 4: Debug Failures",
        "Track and analyze failures"
    )
    
    print_step(1, "Create agent that might fail")
    
    # Define input schema
    class RiskyOperationInput(BaseModel):
        """Input for risky operation tool."""
        input_str: str = Field(description="Input string for the risky operation")
    
    def risky_operation(input_str: str) -> str:
        """An operation that might fail."""
        if "error" in input_str.lower():
            return f"Error: Intentional error for debugging demo"
        return f"Success: {input_str}"
    
    tools = [
        StructuredTool.from_function(
            func=risky_operation,
            name="risky_operation",
            description="A risky operation (use with caution). Input should be a string.",
            args_schema=RiskyOperationInput
        )
    ]
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant. Try to use tools even if they might fail."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    print_step(2, "Track errors with callback")
    
    class ErrorTrackingCallback(StdOutCallbackHandler):
        """Track errors for debugging."""
        
        def __init__(self):
            super().__init__()
            self.errors = []
        
        def on_tool_error(self, error: Exception, **kwargs):
            """Capture tool errors."""
            error_info = {
                "error": str(error),
                "type": type(error).__name__,
                "tool": kwargs.get('name', 'unknown')
            }
            self.errors.append(error_info)
            print(f"\n   ❌ ERROR CAPTURED:")
            print(f"      Type: {error_info['type']}")
            print(f"      Tool: {error_info['tool']}")
            print(f"      Message: {error_info['error']}")
        
        def on_chain_error(self, error: Exception, **kwargs):
            """Capture chain errors."""
            print(f"\n   ❌ CHAIN ERROR: {error}")
    
    error_callback = ErrorTrackingCallback()
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=[error_callback],
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    print_step(3, "Test with inputs that cause errors")
    
    test_inputs = [
        "Use the risky operation with input: 'test'",
        "Use the risky operation with input: 'trigger error'",
    ]
    
    for test_input in test_inputs:
        print(f"\n   🧪 Testing: {test_input}")
        try:
            result = agent_executor.invoke({"input": test_input})
            print(f"   ✅ Result: {result['output'][:100]}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print_step(4, "Error summary")
    
    if error_callback.errors:
        print(f"\n   Found {len(error_callback.errors)} errors:")
        for i, err in enumerate(error_callback.errors, 1):
            print(f"\n   Error {i}:")
            print(f"      Tool: {err['tool']}")
            print(f"      Type: {err['type']}")
            print(f"      Message: {err['error']}")
    else:
        print("\n   ✅ No errors encountered")


def example_5_performance_metrics():
    """
    Example 5: Performance Metrics
    Measure and optimize performance
    """
    print_section(
        "Example 5: Performance Metrics",
        "Analyze performance characteristics"
    )
    
    print_step(1, "Create performance tracking")
    
    class PerformanceCallback(StdOutCallbackHandler):
        """Track performance metrics."""
        
        def __init__(self):
            super().__init__()
            self.metrics = {
                "llm_calls": 0,
                "llm_times": [],
                "tool_calls": 0,
                "tool_times": [],
                "total_time": 0
            }
            self.current_start = None
        
        def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
            """Start LLM timer."""
            self.current_start = time.time()
            self.metrics["llm_calls"] += 1
        
        def on_llm_end(self, response: Any, **kwargs):
            """End LLM timer."""
            if self.current_start:
                duration = time.time() - self.current_start
                self.metrics["llm_times"].append(duration)
        
        def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
            """Start tool timer."""
            self.current_start = time.time()
            self.metrics["tool_calls"] += 1
        
        def on_tool_end(self, output: str, **kwargs):
            """End tool timer."""
            if self.current_start:
                duration = time.time() - self.current_start
                self.metrics["tool_times"].append(duration)
        
        def print_metrics(self):
            """Print performance metrics."""
            print("\n" + "="*70)
            print("   ⚡ PERFORMANCE METRICS")
            print("="*70)
            
            print(f"\n   LLM Calls: {self.metrics['llm_calls']}")
            if self.metrics['llm_times']:
                avg_llm = sum(self.metrics['llm_times']) / len(self.metrics['llm_times'])
                print(f"   Avg LLM Time: {avg_llm:.3f}s")
                print(f"   Min LLM Time: {min(self.metrics['llm_times']):.3f}s")
                print(f"   Max LLM Time: {max(self.metrics['llm_times']):.3f}s")
            
            print(f"\n   Tool Calls: {self.metrics['tool_calls']}")
            if self.metrics['tool_times']:
                avg_tool = sum(self.metrics['tool_times']) / len(self.metrics['tool_times'])
                print(f"   Avg Tool Time: {avg_tool:.3f}s")
            
            total = sum(self.metrics['llm_times']) + sum(self.metrics['tool_times'])
            print(f"\n   Total Time: {total:.3f}s")
            print("="*70 + "\n")
    
    print_step(2, "Run agent with performance tracking")
    
    # Define input schema
    class SlowCalculatorInput(BaseModel):
        """Input for slow calculator tool."""
        expression: str = Field(description="Mathematical expression to evaluate")
    
    def slow_calculation(expression: str) -> str:
        """Simulate slow calculation."""
        time.sleep(0.5)  # Simulate work
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Error in calculation"
    
    tools = [
        StructuredTool.from_function(
            func=slow_calculation,
            name="calculator",
            description="Calculate expressions. Input should be a valid math expression.",
            args_schema=SlowCalculatorInput
        )
    ]
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a calculator assistant."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    perf_callback = PerformanceCallback()
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        callbacks=[perf_callback],
        handle_parsing_errors=True
    )
    
    print("\n   Executing agent...")
    start = time.time()
    
    result = agent_executor.invoke({
        "input": "Calculate 10 + 20, then 30 * 2"
    })
    
    total_time = time.time() - start
    perf_callback.metrics["total_time"] = total_time
    
    print_response(result["output"], "Result")
    
    print_step(3, "Performance analysis")
    perf_callback.print_metrics()
    
    print_step(4, "Optimization recommendations")
    
    recommendations = """
    💡 Performance Optimization Tips:
    
    1. Reduce LLM Calls
       - Use smaller models for simple tasks
       - Cache frequent queries
       - Batch similar requests
    
    2. Optimize Tool Execution
       - Use async tools when possible
       - Implement caching
       - Parallelize independent operations
    
    3. Prompt Engineering
       - Shorter prompts = faster responses
       - Be specific to reduce reasoning time
       - Use examples efficiently
    
    4. Infrastructure
       - Use faster API endpoints
       - Implement request pooling
       - Consider local models for latency
    """
    
    print(recommendations)


def example_6_production_monitoring():
    """
    Example 6: Production Monitoring Setup
    Complete monitoring for production
    """
    print_section(
        "Example 6: Production Monitoring",
        "Set up comprehensive monitoring"
    )
    
    guide = """
    🏭 Production Monitoring Checklist
    
    1. ✅ BASIC SETUP
       [ ] LangSmith account created
       [ ] API keys configured
       [ ] Project structure defined
       [ ] Environment variables set
    
    2. ✅ TRACING
       [ ] All chains traced
       [ ] All agents traced
       [ ] Custom callbacks implemented
       [ ] Trace sampling configured (if needed)
    
    3. ✅ METRICS
       [ ] Token usage tracked
       [ ] Cost monitoring enabled
       [ ] Latency tracked
       [ ] Error rates monitored
       [ ] Success rates tracked
    
    4. ✅ ALERTS
       [ ] Error rate alerts
       [ ] Cost threshold alerts
       [ ] Latency alerts
       [ ] Failure alerts
       [ ] Anomaly detection
    
    5. ✅ DASHBOARDS
       [ ] Real-time metrics dashboard
       [ ] Cost analytics
       [ ] Performance trends
       [ ] User behavior insights
       [ ] Quality metrics
    
    6. ✅ DEBUGGING
       [ ] Error logging configured
       [ ] Stack traces captured
       [ ] Request replay enabled
       [ ] Prompt versioning
       [ ] A/B test tracking
    
    7. ✅ OPTIMIZATION
       [ ] Baseline metrics established
       [ ] Regular performance reviews
       [ ] Cost optimization ongoing
       [ ] Quality improvements tracked
       [ ] User feedback collected
    
    8. ✅ SECURITY
       [ ] PII filtering enabled
       [ ] Sensitive data masked
       [ ] Access controls configured
       [ ] Audit logs enabled
       [ ] Compliance verified
    
    Integration Example:
    
    ```python
    import os
    from langsmith import Client
    
    # Configure LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "your_key"
    os.environ["LANGCHAIN_PROJECT"] = "production"
    
    # Create client for custom metrics
    client = Client()
    
    # Log custom events
    client.create_run(
        name="custom_operation",
        inputs={"query": "..."},
        outputs={"result": "..."},
        run_type="chain"
    )
    
    # Add feedback
    client.create_feedback(
        run_id=run.id,
        key="user_rating",
        score=5.0
    )
    ```
    
    Best Practices:
    - Monitor continuously, not just during deployment
    - Set up alerts before issues occur
    - Review traces regularly
    - Optimize based on data, not assumptions
    - Keep historical data for trends
    - Document your monitoring setup
    """
    
    print(guide)
    
    print("\n" + "="*70)
    print("   📊 Monitoring URLs")
    print("="*70)
    print("   LangSmith Dashboard: https://smith.langchain.com/")
    print("   Documentation: https://docs.smith.langchain.com/")
    print("="*70 + "\n")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" LANGSMITH MONITORING - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_enable_tracing()
        input("\nPress Enter to continue to next example...")
        
        example_2_monitor_agent()
        input("\nPress Enter to continue to next example...")
        
        example_3_track_costs()
        input("\nPress Enter to continue to next example...")
        
        example_4_debug_failures()
        input("\nPress Enter to continue to next example...")
        
        example_5_performance_metrics()
        input("\nPress Enter to continue to next example...")
        
        example_6_production_monitoring()
        
        print("\n" + "="*70)
        print(" ✅ All monitoring examples completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

