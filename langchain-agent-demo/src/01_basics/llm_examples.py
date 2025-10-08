"""
LLM Interaction Examples
========================

This module demonstrates various ways to interact with LLMs:
- Synchronous vs asynchronous calls
- Streaming responses
- Token usage tracking
- Temperature and parameter tuning
- Batch processing

Learning Objectives:
- Understand different LLM invocation patterns
- Master streaming for better UX
- Optimize token usage
- Tune model parameters
"""

import sys
import asyncio
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from src.utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


def example_1_basic_invocation():
    """
    Example 1: Basic LLM Invocation
    Simple synchronous calls to the model
    """
    print_section(
        "Example 1: Basic LLM Invocation",
        "Standard synchronous calls to the language model"
    )
    
    print_step(1, "Create an LLM instance")
    llm = ChatOpenAI(
        model=settings.default_model,
        temperature=0.7
    )
    print(f"   Model: {settings.default_model}")
    print(f"   Temperature: 0.7")
    
    print_step(2, "Invoke with a simple message")
    messages = [
        ("system", "You are a helpful assistant that speaks like a pirate."),
        ("human", "Tell me about artificial intelligence in one sentence.")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({})
    print_response(response, "Pirate AI Explanation")


def example_2_streaming():
    """
    Example 2: Streaming Responses
    Stream tokens as they're generated for better UX
    """
    print_section(
        "Example 2: Streaming Responses",
        "Stream tokens in real-time for immediate feedback"
    )
    
    print_step(1, "Enable streaming mode")
    llm = ChatOpenAI(
        model=settings.default_model,
        temperature=0.7,
        streaming=True
    )
    
    print_step(2, "Stream a response")
    prompt = ChatPromptTemplate.from_template(
        "Write a short story (3-4 sentences) about {topic}."
    )
    chain = prompt | llm | StrOutputParser()
    
    print("\n   📝 Streaming response:\n   ", end="")
    
    for chunk in chain.stream({"topic": "a robot learning to paint"}):
        print(chunk, end="", flush=True)
    
    print("\n")


def example_3_token_tracking():
    """
    Example 3: Token Usage Tracking
    Monitor token consumption and costs
    """
    print_section(
        "Example 3: Token Usage Tracking",
        "Track token usage and estimate costs"
    )
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "Explain {concept} in exactly {length} words."
    )
    chain = prompt | llm | StrOutputParser()
    
    print_step(1, "Make a request with callback tracking")
    
    with get_openai_callback() as cb:
        response = chain.invoke({
            "concept": "quantum computing",
            "length": "50"
        })
        
        print(f"\n   📊 Token Usage Statistics:")
        print(f"      Prompt tokens: {cb.prompt_tokens}")
        print(f"      Completion tokens: {cb.completion_tokens}")
        print(f"      Total tokens: {cb.total_tokens}")
        print(f"      Total cost: ${cb.total_cost:.6f}")
    
    print_response(response, "Response")


def example_4_temperature_comparison():
    """
    Example 4: Temperature Comparison
    See how temperature affects creativity and randomness
    """
    print_section(
        "Example 4: Temperature Comparison",
        "Compare responses at different temperature settings"
    )
    
    prompt = ChatPromptTemplate.from_template(
        "Complete this sentence creatively: 'The future of technology is...'"
    )
    
    temperatures = [0.0, 0.5, 1.0, 1.5]
    
    print_step(1, "Generate responses at different temperatures")
    
    for temp in temperatures:
        llm = ChatOpenAI(model=settings.default_model, temperature=temp)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({})
        
        print(f"\n   🌡️  Temperature: {temp}")
        print(f"   Response: {response}")


def example_5_async_invocation():
    """
    Example 5: Asynchronous Invocation
    Use async for concurrent operations
    """
    print_section(
        "Example 5: Asynchronous Invocation",
        "Process multiple requests concurrently"
    )
    
    print_step(1, "Define async function")
    
    async def generate_completion(topic: str) -> str:
        """Generate a completion asynchronously."""
        llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
        prompt = ChatPromptTemplate.from_template(
            "Write one interesting fact about {topic}."
        )
        chain = prompt | llm | StrOutputParser()
        return await chain.ainvoke({"topic": topic})
    
    print_step(2, "Process multiple topics concurrently")
    
    async def process_batch():
        topics = ["Python", "JavaScript", "Rust", "Go", "TypeScript"]
        print(f"   Topics: {', '.join(topics)}\n")
        
        # Process all topics concurrently
        tasks = [generate_completion(topic) for topic in topics]
        results = await asyncio.gather(*tasks)
        
        for topic, result in zip(topics, results):
            print(f"   📚 {topic}: {result}\n")
    
    # Run the async function
    asyncio.run(process_batch())


def example_6_batch_processing():
    """
    Example 6: Batch Processing
    Efficiently process multiple inputs
    """
    print_section(
        "Example 6: Batch Processing",
        "Process multiple inputs in a single batch"
    )
    
    print_step(1, "Prepare batch inputs")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "Translate '{text}' to {language}."
    )
    chain = prompt | llm | StrOutputParser()
    
    inputs = [
        {"text": "Hello, how are you?", "language": "Spanish"},
        {"text": "Hello, how are you?", "language": "French"},
        {"text": "Hello, how are you?", "language": "German"},
        {"text": "Hello, how are you?", "language": "Japanese"},
    ]
    
    print_step(2, "Process batch")
    
    with get_openai_callback() as cb:
        results = chain.batch(inputs)
        
        for inp, result in zip(inputs, results):
            print(f"\n   🌍 {inp['language']}: {result}")
        
        print(f"\n   📊 Total tokens used: {cb.total_tokens}")
        print(f"   💰 Total cost: ${cb.total_cost:.6f}")


def example_7_parameter_tuning():
    """
    Example 7: Advanced Parameter Tuning
    Explore other model parameters
    """
    print_section(
        "Example 7: Parameter Tuning",
        "Fine-tune model behavior with various parameters"
    )
    
    print_step(1, "Test different parameter combinations")
    
    prompt = ChatPromptTemplate.from_template(
        "List 5 creative names for a {product_type}."
    )
    
    configs = [
        {"temperature": 0.3, "top_p": 0.5, "description": "Conservative & focused"},
        {"temperature": 0.9, "top_p": 0.9, "description": "Creative & diverse"},
        {"temperature": 1.2, "top_p": 1.0, "description": "Very creative & wild"},
    ]
    
    for config in configs:
        llm = ChatOpenAI(
            model=settings.default_model,
            temperature=config["temperature"],
            top_p=config["top_p"]
        )
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({"product_type": "smart coffee maker"})
        
        print(f"\n   ⚙️  Config: {config['description']}")
        print(f"   Temperature: {config['temperature']}, Top-p: {config['top_p']}")
        print(f"   Result:\n{response}\n")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" LLM INTERACTION PATTERNS - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_basic_invocation()
        input("\nPress Enter to continue to next example...")
        
        example_2_streaming()
        input("\nPress Enter to continue to next example...")
        
        example_3_token_tracking()
        input("\nPress Enter to continue to next example...")
        
        example_4_temperature_comparison()
        input("\nPress Enter to continue to next example...")
        
        example_5_async_invocation()
        input("\nPress Enter to continue to next example...")
        
        example_6_batch_processing()
        input("\nPress Enter to continue to next example...")
        
        example_7_parameter_tuning()
        
        print("\n" + "="*70)
        print(" ✅ All LLM examples completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

