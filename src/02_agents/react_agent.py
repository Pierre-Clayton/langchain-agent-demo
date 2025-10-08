"""
ReAct Agent Examples
====================

This module demonstrates the ReAct (Reasoning + Acting) pattern:
- What is ReAct?
- How ReAct agents reason
- Tool chaining
- Complex multi-step tasks

Learning Objectives:
- Understand the ReAct pattern
- See reasoning before acting
- Handle complex multi-tool workflows
- Error recovery and retrying
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from .custom_tools import get_all_tools
from ..utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


def example_1_react_pattern():
    """
    Example 1: Understanding ReAct
    See the Reasoning -> Action -> Observation loop
    """
    print_section(
        "Example 1: The ReAct Pattern",
        "Reasoning and Acting in a loop"
    )
    
    print("""
    📖 The ReAct Pattern:
    
    1. THOUGHT: Agent reasons about what to do
    2. ACTION: Agent decides which tool to use
    3. OBSERVATION: Agent sees the tool's result
    4. Repeat until the answer is found
    
    This creates a powerful reasoning loop!
    """)
    
    print_step(1, "Set up tools")
    
    # Define input schemas
    class UserInfoInput(BaseModel):
        """Input for user info tool."""
        query: str = Field(default="", description="Query parameter (not used)")
    
    class CalculatorInputReact(BaseModel):
        """Input for calculator tool."""
        expression: str = Field(description="Mathematical expression to evaluate")
    
    # Define tool functions
    def get_user_info(query: str = "") -> str:
        return "User ID: 12345, Username: john_doe, Age: 30, Country: USA"
    
    def calculator_func(expression: str) -> str:
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    tools = [
        StructuredTool.from_function(
            func=get_user_info,
            name="get_user_info",
            description="Get information about a user",
            args_schema=UserInfoInput
        ),
        StructuredTool.from_function(
            func=calculator_func,
            name="calculator",
            description="Calculate mathematical expressions. Input should be a valid math expression like '30 + 10'.",
            args_schema=CalculatorInputReact
        ),
    ]
    
    print_step(2, "Create ReAct agent")
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an agent that uses the ReAct pattern.\n\n"
         "For each step:\n"
         "1. THINK about what information you need\n"
         "2. ACT by using a tool\n"
         "3. OBSERVE the result\n"
         "4. Repeat until you can answer\n\n"
         "Be explicit about your reasoning."),
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
    
    print_step(3, "Watch the ReAct loop in action")
    question = "Get the user's information, then calculate what their age will be in 10 years"
    
    print(f"\n   ❓ Question: {question}\n")
    result = agent_executor.invoke({"input": question})
    
    print_response(result["output"], "Final Answer")


def example_2_complex_workflow():
    """
    Example 2: Complex Multi-Step Workflow
    Chain multiple tools together
    """
    print_section(
        "Example 2: Complex Workflow",
        "Agent handling a multi-step research task"
    )
    
    print_step(1, "Define specialized tools")
    
    # Define input schemas
    class SearchPapersInput(BaseModel):
        """Input for search papers tool."""
        topic: str = Field(description="The research topic to search for")
    
    class SummarizePaperInput(BaseModel):
        """Input for summarize paper tool."""
        title: str = Field(description="The title of the paper to summarize")
    
    class ComparePapersInput(BaseModel):
        """Input for compare papers tool."""
        comparison: str = Field(description="Paper titles to compare in format 'paper1 | paper2'")
    
    # Create custom tools for this example
    def search_papers(topic: str) -> str:
        """Simulate searching for research papers."""
        return f"Found 3 papers on {topic}:\n" \
               f"1. 'Introduction to {topic}' (2023)\n" \
               f"2. 'Advanced {topic} Techniques' (2024)\n" \
               f"3. '{topic} in Practice' (2024)"
    
    def summarize_paper(title: str) -> str:
        """Simulate summarizing a paper."""
        return f"Summary of '{title}':\n" \
               f"This paper discusses key concepts and presents novel approaches. " \
               f"Main findings include improved performance and practical applications."
    
    def compare_results(comparison: str) -> str:
        """Compare two papers."""
        parts = comparison.split("|")
        paper1 = parts[0].strip() if len(parts) > 0 else "Paper 1"
        paper2 = parts[1].strip() if len(parts) > 1 else "Paper 2"
        return f"Comparison of {paper1} vs {paper2}:\n" \
               f"Both papers contribute to the field but take different approaches. " \
               f"Paper 1 focuses on theory while Paper 2 emphasizes practical implementation."
    
    tools = [
        StructuredTool.from_function(
            func=search_papers,
            name="search_papers",
            description="Search for academic papers on a topic. Input: topic name",
            args_schema=SearchPapersInput
        ),
        StructuredTool.from_function(
            func=summarize_paper,
            name="summarize_paper",
            description="Get a summary of a paper. Input: paper title",
            args_schema=SummarizePaperInput
        ),
        StructuredTool.from_function(
            func=compare_results,
            name="compare_papers",
            description="Compare two papers. Input: 'paper1_title | paper2_title'",
            args_schema=ComparePapersInput
        ),
    ]
    
    print_step(2, "Create research agent")
    llm = ChatOpenAI(model=settings.default_model, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant that helps users find and analyze academic papers.\n"
         "Break down complex requests into steps:\n"
         "1. Search for relevant papers\n"
         "2. Summarize interesting papers\n"
         "3. Compare or synthesize findings\n\n"
         "Be thorough and methodical."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    print_step(3, "Execute complex research task")
    task = "Find papers about 'neural networks', summarize the most recent one, " \
           "and compare it with the paper about advanced techniques"
    
    print(f"\n   📋 Task: {task}\n")
    result = agent_executor.invoke({"input": task})
    
    print_response(result["output"], "Research Report")


def example_3_error_handling():
    """
    Example 3: Error Handling and Recovery
    How agents handle errors and retry
    """
    print_section(
        "Example 3: Error Handling",
        "Agent recovering from errors and retrying"
    )
    
    print_step(1, "Create tools with potential errors")
    
    # Define input schemas
    class ApiInput(BaseModel):
        """Input for API tools."""
        query: str = Field(description="The query to send to the API")
    
    call_count = {"unstable_api": 0}
    
    def unstable_api(query: str) -> str:
        """Simulate an API that fails sometimes."""
        call_count["unstable_api"] += 1
        
        if call_count["unstable_api"] <= 2:
            # Return error as string so agent can see it and respond
            return f"Error: API temporarily unavailable (attempt {call_count['unstable_api']}). Try again or use backup."
        
        return f"Success! Data for '{query}': [Important information here]"
    
    def reliable_backup(query: str) -> str:
        """Reliable backup tool."""
        return f"Backup data for '{query}': [Cached information]"
    
    tools = [
        StructuredTool.from_function(
            func=unstable_api,
            name="primary_api",
            description="Primary API for getting data (may be unstable)",
            args_schema=ApiInput
        ),
        StructuredTool.from_function(
            func=reliable_backup,
            name="backup_api",
            description="Reliable backup API (use if primary fails)",
            args_schema=ApiInput
        ),
    ]
    
    print_step(2, "Create resilient agent")
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a resilient agent.\n"
         "If a tool fails, try alternative approaches.\n"
         "Don't give up after one failure - be persistent!"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6
    )
    
    print_step(3, "Test error recovery")
    print("\n   The primary API will return errors twice, then succeed on the third attempt\n")
    
    result = agent_executor.invoke({
        "input": "Get data for 'user_profile' from the primary API"
    })
    
    print_response(result["output"], "Result")


def example_4_real_world_scenario():
    """
    Example 4: Real-World Scenario
    Complete task with multiple tools
    """
    print_section(
        "Example 4: Real-World Scenario",
        "Agent completing a practical task"
    )
    
    print_step(1, "Use real custom tools")
    from .custom_tools import (
        create_calculator_tool,
        create_wikipedia_tool,
        create_word_counter_tool
    )
    
    tools = [
        create_calculator_tool(),
        create_wikipedia_tool(),
        create_word_counter_tool(),
    ]
    
    print_step(2, "Create practical agent")
    llm = ChatOpenAI(model=settings.default_model, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that can research topics, perform calculations, "
         "and analyze text. Break down complex tasks into steps and use your tools effectively.\n\n"
         "Important: If a tool returns an error or says it's unavailable, don't keep retrying it. "
         "Instead, use your own knowledge to answer the question and continue with the task."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    print_step(3, "Solve a real-world problem")
    task = (
        "Research what the Python programming language is, "
        "then count how many words are in your explanation, "
        "and finally calculate what 10% of that word count is."
    )
    
    print(f"\n   📋 Task: {task}\n")
    result = agent_executor.invoke({"input": task})
    
    print_response(result["output"], "Complete Solution")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" REACT AGENT EXAMPLES - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_react_pattern()
        input("\nPress Enter to continue to next example...")
        
        example_2_complex_workflow()
        input("\nPress Enter to continue to next example...")
        
        example_3_error_handling()
        input("\nPress Enter to continue to next example...")
        
        example_4_real_world_scenario()
        
        print("\n" + "="*70)
        print(" ✅ All ReAct examples completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

