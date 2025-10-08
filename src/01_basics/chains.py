"""
Basic Chain Examples
====================

This module demonstrates the fundamental building blocks of LangChain:
- Simple chains (Prompt -> LLM -> Output)
- Sequential chains (multiple steps)
- LCEL (LangChain Expression Language)
- Output parsing

Learning Objectives:
- Understand what chains are and why they're useful
- Learn how to compose chains using LCEL
- Parse and transform outputs
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from src.utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


class Joke(BaseModel):
    """Schema for a structured joke."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    rating: int = Field(description="Funniness rating from 1-10")


def example_1_simple_chain():
    """
    Example 1: Simple Chain
    The most basic chain: Prompt -> LLM -> Output
    """
    print_section(
        "Example 1: Simple Chain",
        "A basic chain that tells a joke about a given topic"
    )
    
    # Step 1: Create a prompt template
    print_step(1, "Create a prompt template")
    prompt = ChatPromptTemplate.from_template(
        "Tell me a short, funny joke about {topic}."
    )
    print(f"   Template: {prompt.messages[0].prompt.template}")
    
    # Step 2: Create an LLM
    print_step(2, "Initialize the LLM")
    llm = ChatOpenAI(model=settings.default_model, temperature=0.9)
    print(f"   Model: {settings.default_model}")
    
    # Step 3: Create an output parser
    print_step(3, "Add output parser")
    output_parser = StrOutputParser()
    
    # Step 4: Chain them together using LCEL (|operator)
    print_step(4, "Chain components together using LCEL")
    chain = prompt | llm | output_parser
    print("   Chain: prompt | llm | output_parser")
    
    # Step 5: Invoke the chain
    print_step(5, "Invoke the chain")
    topic = "artificial intelligence"
    response = chain.invoke({"topic": topic})
    
    print_response(response, f"Joke about {topic}")


def example_2_sequential_chain():
    """
    Example 2: Sequential Chain
    Multiple steps where output of one step feeds into the next
    """
    print_section(
        "Example 2: Sequential Chain",
        "Generate a topic, then write a joke, then rate it"
    )
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.8)
    
    # Chain 1: Generate a random topic
    print_step(1, "First chain: Generate a random tech topic")
    topic_prompt = ChatPromptTemplate.from_template(
        "Generate one random technology topic (just the name, nothing else)."
    )
    topic_chain = topic_prompt | llm | StrOutputParser()
    
    # Chain 2: Write a joke about the topic
    print_step(2, "Second chain: Write a joke about the topic")
    joke_prompt = ChatPromptTemplate.from_template(
        "Write a clever joke about: {topic}"
    )
    joke_chain = joke_prompt | llm | StrOutputParser()
    
    # Chain 3: Rate the joke
    print_step(3, "Third chain: Rate the joke")
    rating_prompt = ChatPromptTemplate.from_template(
        "Rate this joke from 1-10 and explain why:\n\n{joke}\n\n"
        "Format: Rating: X/10 - Explanation"
    )
    rating_chain = rating_prompt | llm | StrOutputParser()
    
    # Execute the sequential chain
    print_step(4, "Execute the full pipeline")
    
    # Step 1: Get topic
    topic = topic_chain.invoke({})
    print(f"   📌 Generated topic: {topic}")
    
    # Step 2: Get joke
    joke = joke_chain.invoke({"topic": topic})
    print(f"   😄 Generated joke: {joke[:100]}...")
    
    # Step 3: Get rating
    rating = rating_chain.invoke({"joke": joke})
    
    print_response(rating, "Joke Rating")


def example_3_structured_output():
    """
    Example 3: Structured Output with JSON
    Parse LLM output into structured data using Pydantic models
    """
    print_section(
        "Example 3: Structured Output",
        "Parse LLM output into a structured JSON format"
    )
    
    # Step 1: Define the output structure
    print_step(1, "Define output schema using Pydantic")
    print("   Schema: {setup: str, punchline: str, rating: int}")
    
    # Step 2: Create a JSON output parser
    print_step(2, "Create JSON parser")
    parser = JsonOutputParser(pydantic_object=Joke)
    
    # Step 3: Create prompt with format instructions
    print_step(3, "Create prompt with format instructions")
    prompt = ChatPromptTemplate.from_template(
        "Generate a joke about {topic}.\n\n{format_instructions}"
    )
    
    # Step 4: Build the chain
    print_step(4, "Build the chain")
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    chain = prompt | llm | parser
    
    # Step 5: Invoke with format instructions
    print_step(5, "Invoke the chain")
    result = chain.invoke({
        "topic": "machine learning",
        "format_instructions": parser.get_format_instructions()
    })
    
    # Display structured output
    print_response(
        f"Setup: {result['setup']}\n\n"
        f"Punchline: {result['punchline']}\n\n"
        f"Rating: {result['rating']}/10",
        "Structured Joke"
    )
    print(f"\n   Type: {type(result)}")
    print(f"   Keys: {list(result.keys())}")


def example_4_chain_with_multiple_inputs():
    """
    Example 4: Chain with Multiple Inputs
    Handling multiple variables and RunnablePassthrough
    """
    print_section(
        "Example 4: Multiple Inputs",
        "Create a chain that uses multiple input variables"
    )
    
    print_step(1, "Create a prompt with multiple variables")
    prompt = ChatPromptTemplate.from_template(
        "You are a {role}. Write a {length} message about {topic} "
        "in the style of {style}."
    )
    
    print_step(2, "Build and invoke the chain")
    llm = ChatOpenAI(model=settings.default_model, temperature=0.8)
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "role": "tech blogger",
        "length": "short",
        "topic": "the future of AI",
        "style": "optimistic and inspiring"
    })
    
    print_response(result, "Generated Message")


def example_5_chain_composition():
    """
    Example 5: Advanced Chain Composition
    Using RunnablePassthrough to create complex data flows
    """
    print_section(
        "Example 5: Chain Composition",
        "Advanced composition with data transformation"
    )
    
    print_step(1, "Create multiple chain components")
    
    # Component 1: Generate a company name
    company_prompt = ChatPromptTemplate.from_template(
        "Generate a creative company name for: {industry}"
    )
    llm = ChatOpenAI(model=settings.default_model, temperature=0.9)
    company_chain = company_prompt | llm | StrOutputParser()
    
    # Component 2: Generate a slogan using the company name
    slogan_prompt = ChatPromptTemplate.from_template(
        "Create a catchy slogan for a company called '{company_name}' "
        "in the {industry} industry."
    )
    slogan_chain = slogan_prompt | llm | StrOutputParser()
    
    print_step(2, "Compose chains using RunnablePassthrough")
    
    # Compose: pass industry through and add company_name
    full_chain = (
        {
            "industry": lambda x: x,
            "company_name": lambda x: company_chain.invoke({"industry": x})
        }
        | slogan_chain
    )
    
    print_step(3, "Execute the composed chain")
    industry = "sustainable technology"
    result = full_chain.invoke(industry)
    
    print_response(result, f"Slogan for {industry} company")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" BASIC LANGCHAIN CHAINS - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_simple_chain()
        input("\nPress Enter to continue to next example...")
        
        example_2_sequential_chain()
        input("\nPress Enter to continue to next example...")
        
        example_3_structured_output()
        input("\nPress Enter to continue to next example...")
        
        example_4_chain_with_multiple_inputs()
        input("\nPress Enter to continue to next example...")
        
        example_5_chain_composition()
        
        print("\n" + "="*70)
        print(" ✅ All examples completed successfully!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

