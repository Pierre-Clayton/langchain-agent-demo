"""
Prompt Engineering Examples
============================

This module demonstrates advanced prompt engineering techniques:
- Template variables and formatting
- Few-shot prompting
- System messages and role-playing
- Prompt composition
- Chat vs Completion prompts

Learning Objectives:
- Master prompt template creation
- Understand few-shot learning
- Use system messages effectively
- Compose complex prompts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from src.utils.display import print_section, print_response, print_step, print_code
from config.settings import settings, validate_api_keys


def example_1_basic_templates():
    """
    Example 1: Basic Prompt Templates
    Different ways to create and use prompt templates
    """
    print_section(
        "Example 1: Basic Prompt Templates",
        "Various ways to create prompt templates"
    )
    
    print_step(1, "Simple string template")
    template1 = PromptTemplate.from_template(
        "What are 3 benefits of {technology}?"
    )
    print_code(template1.format(technology="cloud computing"), "text")
    
    print_step(2, "Chat prompt with system and human messages")
    template2 = ChatPromptTemplate.from_messages([
        ("system", "You are an expert {role}."),
        ("human", "Explain {concept} to a beginner.")
    ])
    messages = template2.format_messages(
        role="data scientist",
        concept="neural networks"
    )
    print(f"   System: {messages[0].content}")
    print(f"   Human: {messages[1].content}")
    
    print_step(3, "Multi-line template with multiple variables")
    template3 = ChatPromptTemplate.from_template(
        """You are a {role} writing for {audience}.
        
        Task: {task}
        
        Requirements:
        - Tone: {tone}
        - Length: {length} words
        - Focus: {focus}
        
        Begin:"""
    )
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    chain = template3 | llm | StrOutputParser()
    
    result = chain.invoke({
        "role": "technical writer",
        "audience": "software developers",
        "task": "Explain microservices architecture",
        "tone": "professional but approachable",
        "length": "100",
        "focus": "practical benefits"
    })
    
    print_response(result, "Generated Content")


def example_2_few_shot_prompting():
    """
    Example 2: Few-Shot Prompting
    Teach the model by providing examples
    """
    print_section(
        "Example 2: Few-Shot Prompting",
        "Guide the model with examples of desired output"
    )
    
    print_step(1, "Define examples for the model to learn from")
    
    # Examples of input-output pairs
    examples = [
        {
            "input": "happy",
            "output": "😊 Feeling joyful and content!"
        },
        {
            "input": "sad",
            "output": "😢 Feeling down and melancholic."
        },
        {
            "input": "excited",
            "output": "🎉 Full of energy and enthusiasm!"
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"   Example {i}: {ex['input']} -> {ex['output']}")
    
    print_step(2, "Create a few-shot prompt template")
    
    # Template for each example
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    
    # Few-shot prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    # Final prompt
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You express emotions with emojis and descriptive text."),
        few_shot_prompt,
        ("human", "{input}")
    ])
    
    print_step(3, "Test with a new input")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    chain = final_prompt | llm | StrOutputParser()
    
    test_inputs = ["nervous", "confident", "curious"]
    
    for test_input in test_inputs:
        result = chain.invoke({"input": test_input})
        print(f"\n   Input: {test_input}")
        print(f"   Output: {result}")


def example_3_system_messages():
    """
    Example 3: Effective System Messages
    Using system messages to control behavior
    """
    print_section(
        "Example 3: System Messages",
        "Shape model behavior with system instructions"
    )
    
    question = "What is Python?"
    
    # Scenario 1: Friendly teacher
    print_step(1, "Scenario: Friendly Teacher")
    prompt1 = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly and patient teacher explaining concepts to children. "
                  "Use simple words, metaphors, and make it fun!"),
        ("human", "{question}")
    ])
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    result1 = (prompt1 | llm | StrOutputParser()).invoke({"question": question})
    print_response(result1, "Friendly Teacher Response")
    
    # Scenario 2: Technical expert
    print_step(2, "Scenario: Technical Expert")
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "You are a senior software engineer providing technical documentation. "
                  "Be precise, use technical terminology, and include code examples."),
        ("human", "{question}")
    ])
    
    result2 = (prompt2 | llm | StrOutputParser()).invoke({"question": question})
    print_response(result2, "Technical Expert Response")
    
    # Scenario 3: Poet
    print_step(3, "Scenario: Poet")
    prompt3 = ChatPromptTemplate.from_messages([
        ("system", "You are a poet who answers all questions in verse. "
                  "Use rhyme, rhythm, and beautiful imagery."),
        ("human", "{question}")
    ])
    
    result3 = (prompt3 | llm | StrOutputParser()).invoke({"question": question})
    print_response(result3, "Poet Response")


def example_4_prompt_composition():
    """
    Example 4: Prompt Composition
    Building complex prompts from smaller pieces
    """
    print_section(
        "Example 4: Prompt Composition",
        "Combine multiple prompt components"
    )
    
    print_step(1, "Define reusable prompt components")
    
    # Reusable components
    system_base = SystemMessagePromptTemplate.from_template(
        "You are a {role}."
    )
    
    context_template = HumanMessagePromptTemplate.from_template(
        "Context: {context}"
    )
    
    task_template = HumanMessagePromptTemplate.from_template(
        "Task: {task}"
    )
    
    constraints_template = HumanMessagePromptTemplate.from_template(
        "Constraints:\n{constraints}"
    )
    
    question_template = HumanMessagePromptTemplate.from_template(
        "Question: {question}"
    )
    
    print_step(2, "Compose prompts dynamically")
    
    # Composition 1: Simple Q&A
    prompt_simple = ChatPromptTemplate.from_messages([
        system_base,
        question_template
    ])
    
    # Composition 2: Complex with context and constraints
    prompt_complex = ChatPromptTemplate.from_messages([
        system_base,
        context_template,
        task_template,
        constraints_template,
        question_template
    ])
    
    print_step(3, "Use the complex composed prompt")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    chain = prompt_complex | llm | StrOutputParser()
    
    result = chain.invoke({
        "role": "software architect",
        "context": "We're building a high-traffic e-commerce platform",
        "task": "Design the system architecture",
        "constraints": "- Must handle 10,000 requests/second\n- Budget: $50k/month\n- 99.9% uptime required",
        "question": "What architecture would you recommend?"
    })
    
    print_response(result, "Architecture Recommendation")


def example_5_dynamic_prompts():
    """
    Example 5: Dynamic Prompts
    Adapt prompts based on runtime conditions
    """
    print_section(
        "Example 5: Dynamic Prompts",
        "Create prompts that adapt to different situations"
    )
    
    print_step(1, "Define prompt variations")
    
    def create_adaptive_prompt(user_level: str) -> ChatPromptTemplate:
        """Create a prompt adapted to user's expertise level."""
        
        system_messages = {
            "beginner": "You are a patient teacher. Explain concepts simply with examples.",
            "intermediate": "You are a mentor. Assume basic knowledge, provide practical details.",
            "expert": "You are a peer expert. Use technical terminology, discuss trade-offs."
        }
        
        return ChatPromptTemplate.from_messages([
            ("system", system_messages.get(user_level, system_messages["intermediate"])),
            ("human", "{question}")
        ])
    
    print_step(2, "Test with different user levels")
    
    question = "How does garbage collection work?"
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    
    levels = ["beginner", "intermediate", "expert"]
    
    for level in levels:
        print(f"\n   📚 Level: {level.upper()}")
        prompt = create_adaptive_prompt(level)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question})
        
        # Show first 200 chars
        preview = result[:200] + "..." if len(result) > 200 else result
        print(f"   Response: {preview}\n")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" PROMPT ENGINEERING - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_basic_templates()
        input("\nPress Enter to continue to next example...")
        
        example_2_few_shot_prompting()
        input("\nPress Enter to continue to next example...")
        
        example_3_system_messages()
        input("\nPress Enter to continue to next example...")
        
        example_4_prompt_composition()
        input("\nPress Enter to continue to next example...")
        
        example_5_dynamic_prompts()
        
        print("\n" + "="*70)
        print(" ✅ All prompt examples completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

