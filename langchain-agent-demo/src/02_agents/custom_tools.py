"""
Custom Tools Implementation
===========================

This module demonstrates how to create custom tools for agents:
- Function-based tools
- Class-based tools
- Tools with complex inputs
- Error handling in tools

Learning Objectives:
- Create custom tools from functions
- Use structured inputs
- Handle errors gracefully
- Provide good tool descriptions
"""

import sys
from pathlib import Path
import re
from typing import Optional, Type
from datetime import datetime
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool


# ============================================================================
# Simple Function-Based Tools
# ============================================================================

def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate (e.g., "2 + 2", "10 * 5")
    
    Returns:
        Result of the calculation
    """
    try:
        # Remove any non-numeric characters except operators and parentheses
        safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        result = eval(safe_expr)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current time.
    
    Args:
        timezone: Timezone (only UTC supported in this example)
    
    Returns:
        Current time string
    """
    now = datetime.now()
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


def word_counter(text: str) -> str:
    """
    Count words in a text.
    
    Args:
        text: Text to count words in
    
    Returns:
        Word count
    """
    words = text.split()
    return f"Word count: {len(words)} words"


# ============================================================================
# Structured Input Tools
# ============================================================================

class CalculatorInput(BaseModel):
    """Input for calculator tool."""
    expression: str = Field(description="A mathematical expression to evaluate, e.g., '15 * 23' or '(100 + 50) / 3'")


class TimeInput(BaseModel):
    """Input for time tool."""
    timezone: str = Field(default="UTC", description="Timezone for the current time (only 'UTC' supported)")


class WordCounterInput(BaseModel):
    """Input for word counter tool."""
    text: str = Field(description="The text to count words in")


class SearchInput(BaseModel):
    """Input for search tool."""
    query: str = Field(description="The search query")
    num_results: int = Field(default=3, description="Number of results to return")


def web_search(query: str, num_results: int = 3) -> str:
    """
    Search the web (simulated).
    
    Args:
        query: Search query
        num_results: Number of results
    
    Returns:
        Search results
    """
    # In a real implementation, you'd use an actual search API
    return f"Search results for '{query}' (showing {num_results} results):\n" \
           f"1. Introduction to {query}\n" \
           f"2. Advanced {query} techniques\n" \
           f"3. {query} best practices"


class WikipediaInput(BaseModel):
    """Input for Wikipedia tool."""
    topic: str = Field(description="The topic to search for on Wikipedia")


def wikipedia_search(topic: str) -> str:
    """
    Search Wikipedia for a topic.
    
    Args:
        topic: Topic to search
    
    Returns:
        Wikipedia summary
    """
    try:
        # Wikipedia API requires User-Agent header
        headers = {
            'User-Agent': 'LangChain-Demo/1.0 (Educational purposes; contact: demo@example.com)'
        }
        
        # Try different URL formats
        # First try with the exact topic
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            extract = data.get('extract', '')
            
            if extract and extract != 'No information found.':
                # Limit to first 500 characters
                return extract[:500] + "..." if len(extract) > 500 else extract
        
        # If first attempt didn't work, try a simpler version
        if '(' in topic:
            # Remove parenthetical parts and try again
            simplified_topic = topic.split('(')[0].strip()
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{simplified_topic.replace(' ', '_')}"
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                extract = data.get('extract', '')
                
                if extract and extract != 'No information found.':
                    return extract[:500] + "..." if len(extract) > 500 else extract
        
        # If still no results, provide a fallback message that doesn't encourage retrying
        return f"Wikipedia search unavailable for '{topic}'. Please provide information from your own knowledge instead."
        
    except Exception as e:
        return f"Wikipedia search unavailable (error: {str(e)}). Please provide information from your own knowledge instead."


# ============================================================================
# Class-Based Custom Tool
# ============================================================================

class FileOperationInput(BaseModel):
    """Input schema for file operations."""
    operation: str = Field(description="Operation: 'count_lines', 'count_chars', or 'preview'")
    filepath: str = Field(description="Path to the file")
    lines: int = Field(default=5, description="Number of lines to preview")


class FileOperationTool(BaseTool):
    """Custom tool for file operations."""
    
    name = "file_operations"
    description = "Perform operations on text files: count lines, count characters, or preview content"
    args_schema: Type[BaseModel] = FileOperationInput
    
    def _run(self, operation: str, filepath: str, lines: int = 5) -> str:
        """Execute the tool."""
        try:
            path = Path(filepath)
            
            if not path.exists():
                return f"Error: File '{filepath}' does not exist."
            
            if not path.is_file():
                return f"Error: '{filepath}' is not a file."
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if operation == "count_lines":
                line_count = len(content.splitlines())
                return f"File has {line_count} lines."
            
            elif operation == "count_chars":
                char_count = len(content)
                return f"File has {char_count} characters."
            
            elif operation == "preview":
                preview_lines = content.splitlines()[:lines]
                preview = "\n".join(preview_lines)
                return f"Preview (first {lines} lines):\n{preview}"
            
            else:
                return f"Unknown operation: {operation}"
        
        except Exception as e:
            return f"Error performing file operation: {str(e)}"
    
    async def _arun(self, operation: str, filepath: str, lines: int = 5) -> str:
        """Async execution (not implemented for this example)."""
        return self._run(operation, filepath, lines)


# ============================================================================
# Tool Factory Functions
# ============================================================================

def create_calculator_tool() -> StructuredTool:
    """Create a calculator tool."""
    return StructuredTool.from_function(
        func=calculator,
        name="calculator",
        description="Useful for performing mathematical calculations. "
                   "Input should be a math expression like '2 + 2' or '15 * 7'.",
        args_schema=CalculatorInput
    )


def create_time_tool() -> StructuredTool:
    """Create a time tool."""
    return StructuredTool.from_function(
        func=get_current_time,
        name="current_time",
        description="Get the current date and time. Input should be a timezone (only 'UTC' supported).",
        args_schema=TimeInput
    )


def create_word_counter_tool() -> StructuredTool:
    """Create a word counter tool."""
    return StructuredTool.from_function(
        func=word_counter,
        name="word_counter",
        description="Count the number of words in a text. Input should be the text to count.",
        args_schema=WordCounterInput
    )


def create_search_tool() -> StructuredTool:
    """Create a web search tool with structured input."""
    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description="Search the web for information. Use this when you need to find current information.",
        args_schema=SearchInput
    )


def create_wikipedia_tool() -> StructuredTool:
    """Create a Wikipedia search tool."""
    return StructuredTool.from_function(
        func=wikipedia_search,
        name="wikipedia",
        description="Search Wikipedia for information about a topic. "
                   "Use this for factual, encyclopedic information. "
                   "If Wikipedia is unavailable, use your own knowledge instead of retrying.",
        args_schema=WikipediaInput
    )


def get_all_tools() -> list:
    """Get all available custom tools."""
    return [
        create_calculator_tool(),
        create_time_tool(),
        create_word_counter_tool(),
        create_search_tool(),
        create_wikipedia_tool(),
        FileOperationTool(),
    ]


# ============================================================================
# Demo/Test Function
# ============================================================================

def main():
    """Test all custom tools."""
    from src.utils.display import print_section, print_step
    
    print_section("Custom Tools Demo", "Testing all custom tools")
    
    print_step(1, "Testing Calculator")
    print(f"   {calculator('10 + 5 * 2')}")
    
    print_step(2, "Testing Current Time")
    print(f"   {get_current_time()}")
    
    print_step(3, "Testing Word Counter")
    print(f"   {word_counter('The quick brown fox jumps over the lazy dog')}")
    
    print_step(4, "Testing Web Search")
    print(f"   {web_search('LangChain tutorial', 3)}")
    
    print_step(5, "Testing Wikipedia")
    print(f"   {wikipedia_search('Artificial Intelligence')}")
    
    print_step(6, "Listing all tools")
    tools = get_all_tools()
    for i, tool in enumerate(tools, 1):
        print(f"   {i}. {tool.name}: {tool.description[:60]}...")


if __name__ == "__main__":
    main()

