"""
Helper functions for common operations across examples.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.settings import settings


def load_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = False,
    **kwargs
) -> ChatOpenAI:
    """
    Load and configure an LLM.
    
    Args:
        model: Model name (defaults to settings)
        temperature: Temperature (defaults to settings)
        streaming: Enable streaming
        **kwargs: Additional arguments for ChatOpenAI
    
    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model or settings.default_model,
        temperature=temperature if temperature is not None else settings.temperature,
        streaming=streaming,
        **kwargs
    )


def create_chat_prompt(
    system_message: str,
    human_message: str = "{input}",
    include_history: bool = False
) -> ChatPromptTemplate:
    """
    Create a chat prompt template.
    
    Args:
        system_message: System instruction
        human_message: Human message template
        include_history: Include conversation history
    
    Returns:
        ChatPromptTemplate
    """
    messages = [
        ("system", system_message),
    ]
    
    if include_history:
        messages.append(MessagesPlaceholder(variable_name="chat_history"))
    
    messages.append(("human", human_message))
    
    return ChatPromptTemplate.from_messages(messages)


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_docs(docs) -> str:
    """
    Format documents for display.
    
    Args:
        docs: List of documents
    
    Returns:
        Formatted string
    """
    return "\n\n".join(doc.page_content for doc in docs)

