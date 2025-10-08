"""
Conversation Memory Examples
============================

This module demonstrates how to give agents memory:
- Buffer memory (recent history)
- Window memory (last N messages)
- Summary memory (compressed history)
- Conversation chains

Learning Objectives:
- Add context to conversations
- Manage conversation history
- Handle long conversations
- Maintain state across interactions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


def example_1_buffer_memory():
    """
    Example 1: Buffer Memory
    Store entire conversation history
    """
    print_section(
        "Example 1: Buffer Memory",
        "Store and recall complete conversation history"
    )
    
    print_step(1, "Create conversation chain with buffer memory")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    memory = ConversationBufferMemory()
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    print_step(2, "Have a multi-turn conversation")
    
    exchanges = [
        "Hi! My name is Alice.",
        "What's my name?",
        "I love programming in Python.",
        "What programming language do I like?",
        "Can you remind me what we've talked about?",
    ]
    
    for i, user_input in enumerate(exchanges, 1):
        print(f"\n   👤 User: {user_input}")
        response = conversation.predict(input=user_input)
        print(f"   🤖 Assistant: {response}")
        
        if i < len(exchanges):
            input("      Press Enter to continue...")
    
    print_step(3, "Inspect memory contents")
    print("\n   📝 Memory Buffer:")
    print(f"   {memory.buffer}")


def example_2_window_memory():
    """
    Example 2: Window Memory
    Keep only the last N messages
    """
    print_section(
        "Example 2: Window Memory",
        "Maintain a sliding window of recent messages"
    )
    
    print_step(1, "Create conversation with window size of 2")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    memory = ConversationBufferWindowMemory(k=2)  # Only keep last 2 exchanges
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    print("   (Memory will only keep last 2 exchanges)\n")
    
    print_step(2, "Test memory window")
    
    messages = [
        "Remember this: My favorite color is blue.",
        "Remember this: I have a dog named Max.",
        "Remember this: I work as a teacher.",
        "What's my favorite color?",  # This should be forgotten!
        "What's my dog's name?",      # This should be remembered
        "What's my profession?",      # This should be remembered
    ]
    
    for msg in messages:
        print(f"\n   👤 User: {msg}")
        response = conversation.predict(input=msg)
        print(f"   🤖 Assistant: {response}")
        input("      Press Enter to continue...")


def example_3_summary_memory():
    """
    Example 3: Summary Memory
    Compress long conversations into summaries
    """
    print_section(
        "Example 3: Summary Memory",
        "Compress conversation history to save tokens"
    )
    
    print_step(1, "Create conversation with summary memory")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    memory = ConversationSummaryMemory(llm=llm)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    print_step(2, "Have a longer conversation")
    
    messages = [
        "I'm planning a trip to Japan next summer.",
        "I want to visit Tokyo, Kyoto, and Osaka.",
        "I'm particularly interested in historical temples and modern technology.",
        "My budget is around $3000 for a 10-day trip.",
        "Can you summarize what I've told you about my trip?",
    ]
    
    for msg in messages:
        print(f"\n   👤 User: {msg}")
        response = conversation.predict(input=msg)
        print(f"   🤖 Assistant: {response}")
        input("      Press Enter to continue...")
    
    print_step(3, "View the conversation summary")
    print("\n   📋 Summary:")
    print(f"   {memory.buffer}")


def example_4_summary_buffer_memory():
    """
    Example 4: Summary Buffer Memory
    Combination of summary and buffer
    """
    print_section(
        "Example 4: Summary Buffer Memory",
        "Keep recent messages + summary of older messages"
    )
    
    print_step(1, "Create hybrid memory (summary + buffer)")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100  # Summarize when exceeding this limit
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False  # Less verbose for clarity
    )
    
    print("   (Recent messages kept as-is, older ones summarized)\n")
    
    print_step(2, "Build conversation history")
    
    messages = [
        "I'm a software engineer with 5 years of experience.",
        "I specialize in Python and machine learning.",
        "I'm looking for a new job opportunity.",
        "I prefer remote work if possible.",
        "Tell me about my job preferences based on what I've shared.",
    ]
    
    for msg in messages:
        print(f"\n   👤 User: {msg}")
        response = conversation.predict(input=msg)
        print_response(response, "Assistant")


def example_5_custom_memory_chain():
    """
    Example 5: Custom Memory with LCEL
    Build a custom chain with memory
    """
    print_section(
        "Example 5: Custom Memory Chain",
        "Using LCEL with message history"
    )
    
    print_step(1, "Create custom chain with memory")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the conversation history to provide context-aware responses."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Create message history storage
    store = {}
    
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    # Wrap chain with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    print_step(2, "Have conversations in different sessions")
    
    # Session 1
    print("\n   💬 Session 1:")
    session_1_messages = [
        "Hi, I'm learning about LangChain.",
        "What did I say I was learning about?",
    ]
    
    for msg in session_1_messages:
        print(f"\n      👤 User: {msg}")
        response = chain_with_history.invoke(
            {"input": msg},
            config={"configurable": {"session_id": "session_1"}}
        )
        print(f"      🤖 Assistant: {response}")
    
    # Session 2 (different context)
    print("\n   💬 Session 2 (separate context):")
    session_2_messages = [
        "I'm studying physics.",
        "What am I studying?",
    ]
    
    for msg in session_2_messages:
        print(f"\n      👤 User: {msg}")
        response = chain_with_history.invoke(
            {"input": msg},
            config={"configurable": {"session_id": "session_2"}}
        )
        print(f"      🤖 Assistant: {response}")


def example_6_agent_with_memory():
    """
    Example 6: Agent with Memory
    Combine agents and memory
    """
    print_section(
        "Example 6: Agent with Memory",
        "Give an agent the ability to remember past interactions"
    )
    
    print_step(1, "Create tools and agent with memory")
    
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.tools import StructuredTool
    from langchain.pydantic_v1 import BaseModel, Field
    
    # Define input schemas
    class WeatherInput(BaseModel):
        """Input for weather tool."""
        location: str = Field(description="The location to get weather for")
    
    class ReminderInput(BaseModel):
        """Input for reminder tool."""
        reminder: str = Field(description="The reminder text to set")
    
    # Simple tools
    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny and 72°F."
    
    def set_reminder(reminder: str) -> str:
        return f"Reminder set: {reminder}"
    
    tools = [
        StructuredTool.from_function(
            func=get_weather,
            name="get_weather",
            description="Get weather for a location. Provide the location name as input.",
            args_schema=WeatherInput
        ),
        StructuredTool.from_function(
            func=set_reminder,
            name="set_reminder",
            description="Set a reminder. Provide the reminder text as input.",
            args_schema=ReminderInput
        ),
    ]
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful personal assistant with access to tools. "
                  "Remember what the user tells you."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    
    print_step(2, "Test agent memory across interactions")
    
    queries = [
        "I live in San Francisco. What's the weather like here?",
        "Set a reminder to water the plants.",
        "Where did I say I live?",  # Testing memory
        "What reminder did I set?",  # Testing memory
    ]
    
    for query in queries:
        print(f"\n   👤 User: {query}")
        result = agent_executor.invoke({"input": query})
        print_response(result["output"], "Assistant")
        input("      Press Enter to continue...")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" CONVERSATION MEMORY - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_buffer_memory()
        input("\nPress Enter to continue to next example...")
        
        example_2_window_memory()
        input("\nPress Enter to continue to next example...")
        
        example_3_summary_memory()
        input("\nPress Enter to continue to next example...")
        
        example_4_summary_buffer_memory()
        input("\nPress Enter to continue to next example...")
        
        example_5_custom_memory_chain()
        input("\nPress Enter to continue to next example...")
        
        example_6_agent_with_memory()
        
        print("\n" + "="*70)
        print(" ✅ All memory examples completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

