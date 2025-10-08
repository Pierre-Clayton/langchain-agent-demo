"""
Simple Agent Examples
=====================

This module demonstrates basic agent concepts:
- What is an agent?
- How agents use tools
- Agent reasoning process
- Different agent types

Learning Objectives:
- Understand the agent reasoning loop
- See how agents decide which tools to use
- Learn about tool selection and execution
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from .custom_tools import (
    create_calculator_tool,
    create_time_tool,
    create_word_counter_tool,
    create_wikipedia_tool
)
from ..utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


def example_1_basic_agent():
    """
    Example 1: Basic Agent with Calculator
    Simplest possible agent - just one tool
    """
    print_section(
        "Example 1: Basic Agent",
        "An agent that can perform calculations"
    )
    
    print_step(1, "Create a calculator tool")
    tools = [create_calculator_tool()]
    print(f"   Tool: {tools[0].name}")
    print(f"   Description: {tools[0].description}")
    
    print_step(2, "Create the LLM")
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    print_step(3, "Create the agent prompt")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can perform calculations. "
                  "When you need to calculate something, use the calculator tool with a single mathematical expression. "
                  "For example, for '15 multiplied by 23', use the calculator with '15 * 23'."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    print_step(4, "Create the agent")
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print_step(5, "Run the agent")
    questions = [
        "What is 15 multiplied by 23?",
        "Calculate the result of (100 + 50) divided by 3",
    ]
    
    for question in questions:
        print(f"\n   ❓ Question: {question}")
        result = agent_executor.invoke({"input": question})
        print_response(result["output"], "Answer")


def example_2_multi_tool_agent():
    """
    Example 2: Agent with Multiple Tools
    Agent decides which tool to use
    """
    print_section(
        "Example 2: Multi-Tool Agent",
        "Agent with multiple tools - it decides which to use"
    )
    
    print_step(1, "Create multiple tools")
    tools = [
        create_calculator_tool(),
        create_time_tool(),
        create_word_counter_tool(),
    ]
    
    print("   Available tools:")
    for i, tool in enumerate(tools, 1):
        print(f"      {i}. {tool.name}: {tool.description[:50]}...")
    
    print_step(2, "Create and configure the agent")
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to various tools. "
                  "Use them wisely to answer questions. "
                  "For calculations, use the calculator tool with a single mathematical expression. "
                  "For time questions, use the current_time tool. "
                  "For word counting, use the word_counter tool with the text to count."),
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
    
    print_step(3, "Test the agent with various questions")
    
    questions = [
        "What time is it right now?",
        "How many words are in this sentence: 'LangChain makes building AI applications easy'?",
        "What is 456 plus 789?",
        "First tell me the time, then calculate 25 times 4",
    ]
    
    for question in questions:
        print(f"\n   ❓ Question: {question}")
        result = agent_executor.invoke({"input": question})
        print_response(result["output"], "Answer")
        input("   Press Enter to continue...")


def example_3_research_agent():
    """
    Example 3: Research Agent
    Agent that can look up information
    """
    print_section(
        "Example 3: Research Agent",
        "Agent that can search for information"
    )
    
    print_step(1, "Create research tools")
    tools = [
        create_wikipedia_tool(),
        create_calculator_tool(),
    ]
    
    print_step(2, "Create a research-focused agent")
    llm = ChatOpenAI(model=settings.default_model, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a knowledgeable research assistant. "
         "When asked about facts or topics, use Wikipedia to find accurate information. "
         "When calculations are needed, use the calculator with a single mathematical expression. "
         "Always cite your sources and be precise."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    print_step(3, "Ask research questions")
    
    questions = [
        "Tell me about Python programming language",
        "What is quantum computing? Give me a brief overview.",
    ]
    
    for question in questions:
        print(f"\n   ❓ Question: {question}")
        result = agent_executor.invoke({"input": question})
        print_response(result["output"], "Research Result")
        input("   Press Enter to continue...")


def example_4_agent_reasoning():
    """
    Example 4: Understanding Agent Reasoning
    See how agents think through problems
    """
    print_section(
        "Example 4: Agent Reasoning Process",
        "Watch the agent's thought process in detail"
    )
    
    tools = [
        create_calculator_tool(),
        create_word_counter_tool(),
    ]
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a methodical assistant. "
         "Think step-by-step and explain your reasoning clearly. "
         "For calculations, use the calculator tool with a single mathematical expression. "
         "For word counting, use the word_counter tool with the text to count."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # This shows the reasoning process
        handle_parsing_errors=True,
        return_intermediate_steps=True  # Return the reasoning steps
    )
    
    print_step(1, "Ask a multi-step question")
    question = "Count the words in 'Machine learning is fascinating', then multiply that count by 10"
    
    print(f"\n   ❓ Question: {question}")
    print("\n   🧠 Watch the agent's reasoning:\n")
    
    result = agent_executor.invoke({"input": question})
    
    print_response(result["output"], "Final Answer")
    
    print("\n   📋 Reasoning Steps:")
    for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
        print(f"\n      Step {i}:")
        print(f"         Action: {action.tool}")
        print(f"         Input: {action.tool_input}")
        print(f"         Observation: {observation}")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" SIMPLE AGENT EXAMPLES - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_basic_agent()
        input("\nPress Enter to continue to next example...")
        
        example_2_multi_tool_agent()
        input("\nPress Enter to continue to next example...")
        
        example_3_research_agent()
        input("\nPress Enter to continue to next example...")
        
        example_4_agent_reasoning()
        
        print("\n" + "="*70)
        print(" ✅ All agent examples completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

