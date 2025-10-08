"""
Conditional LangGraph Examples
==============================

This module demonstrates conditional routing in graphs:
- Conditional edges
- Dynamic routing based on state
- Decision nodes
- Complex branching logic

Learning Objectives:
- Add conditional logic to graphs
- Route based on state values
- Create decision nodes
- Build adaptive workflows
"""

import sys
from pathlib import Path
from typing import TypedDict, Literal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from src.utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


def example_1_simple_conditional():
    """
    Example 1: Simple Conditional Routing
    Route based on a single condition
    """
    print_section(
        "Example 1: Simple Conditional",
        "Route to different nodes based on state"
    )
    
    print_step(1, "Define state")
    
    class State(TypedDict):
        number: int
        result: str
    
    print_step(2, "Create processing nodes")
    
    def check_number(state: State) -> State:
        """Classify the number."""
        print(f"   🔍 Checking number: {state['number']}")
        return state
    
    def process_even(state: State) -> State:
        """Process even numbers."""
        print(f"   ✅ Processing as EVEN")
        return {"result": f"{state['number']} is even"}
    
    def process_odd(state: State) -> State:
        """Process odd numbers."""
        print(f"   ✅ Processing as ODD")
        return {"result": f"{state['number']} is odd"}
    
    print_step(3, "Define routing function")
    
    def route_number(state: State) -> Literal["even", "odd"]:
        """Route based on even/odd."""
        if state["number"] % 2 == 0:
            return "even"
        return "odd"
    
    print("""
    Routing logic:
    - If number is even → go to 'even' node
    - If number is odd → go to 'odd' node
    """)
    
    print_step(4, "Build conditional graph")
    
    workflow = StateGraph(State)
    workflow.add_node("check", check_number)
    workflow.add_node("even", process_even)
    workflow.add_node("odd", process_odd)
    
    # Regular edge
    workflow.add_edge(START, "check")
    
    # Conditional edge
    workflow.add_conditional_edges(
        "check",
        route_number,
        {"even": "even", "odd": "odd"}
    )
    
    # Both routes end
    workflow.add_edge("even", END)
    workflow.add_edge("odd", END)
    
    app = workflow.compile()
    
    print_step(5, "Test with different numbers")
    
    for num in [10, 7, 22, 15]:
        print(f"\n   Testing: {num}")
        result = app.invoke({"number": num, "result": ""})
        print(f"   Result: {result['result']}")


def example_2_multi_way_routing():
    """
    Example 2: Multi-Way Routing
    Route to multiple possible destinations
    """
    print_section(
        "Example 2: Multi-Way Routing",
        "Choose from multiple paths based on conditions"
    )
    
    print_step(1, "Create sentiment analysis system")
    
    class SentimentState(TypedDict):
        text: str
        sentiment: str
        response: str
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    def analyze_sentiment(state: SentimentState) -> SentimentState:
        """Analyze text sentiment."""
        print(f"   🔍 Analyzing: '{state['text']}'")
        
        prompt = f"""Analyze the sentiment of this text: "{state['text']}"
        
        Respond with ONLY one word: positive, negative, or neutral"""
        
        sentiment = llm.invoke(prompt).content.strip().lower()
        print(f"   📊 Sentiment detected: {sentiment}")
        
        return {"sentiment": sentiment}
    
    def handle_positive(state: SentimentState) -> SentimentState:
        """Handle positive sentiment."""
        return {"response": "😊 Great to hear positive feedback!"}
    
    def handle_negative(state: SentimentState) -> SentimentState:
        """Handle negative sentiment."""
        return {"response": "😔 We're sorry to hear that. We'll work on improving."}
    
    def handle_neutral(state: SentimentState) -> SentimentState:
        """Handle neutral sentiment."""
        return {"response": "📝 Thank you for your feedback."}
    
    print_step(2, "Define multi-way routing")
    
    def route_by_sentiment(state: SentimentState) -> Literal["positive", "negative", "neutral"]:
        """Route based on sentiment."""
        sentiment = state.get("sentiment", "neutral")
        if "positive" in sentiment:
            return "positive"
        elif "negative" in sentiment:
            return "negative"
        return "neutral"
    
    print_step(3, "Build the graph")
    
    workflow = StateGraph(SentimentState)
    workflow.add_node("analyze", analyze_sentiment)
    workflow.add_node("positive", handle_positive)
    workflow.add_node("negative", handle_negative)
    workflow.add_node("neutral", handle_neutral)
    
    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges(
        "analyze",
        route_by_sentiment,
        {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral"
        }
    )
    
    workflow.add_edge("positive", END)
    workflow.add_edge("negative", END)
    workflow.add_edge("neutral", END)
    
    app = workflow.compile()
    
    print_step(4, "Test with different texts")
    
    texts = [
        "I absolutely love this product! It's amazing!",
        "This is terrible and doesn't work at all.",
        "The product arrived on time.",
    ]
    
    for text in texts:
        print(f"\n   📝 Text: {text}")
        result = app.invoke({"text": text, "sentiment": "", "response": ""})
        print_response(result["response"], "Response")


def example_3_cyclic_workflow():
    """
    Example 3: Cyclic Workflow with Conditions
    Loop back based on conditions
    """
    print_section(
        "Example 3: Cyclic Workflow",
        "Loop until a condition is met"
    )
    
    print_step(1, "Create an iterative improvement system")
    
    class IterativeState(TypedDict):
        content: str
        iteration: int
        max_iterations: int
        is_good: bool
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    
    def improve_content(state: IterativeState) -> IterativeState:
        """Improve the content."""
        print(f"\n   🔄 Iteration {state['iteration'] + 1}")
        
        prompt = f"Improve this text to be more engaging: {state['content']}"
        improved = llm.invoke(prompt).content
        
        return {
            "content": improved,
            "iteration": state["iteration"] + 1
        }
    
    def evaluate_content(state: IterativeState) -> IterativeState:
        """Evaluate if content is good enough."""
        # Simple evaluation: check length and iteration count
        is_good = len(state["content"]) > 400 or state["iteration"] >= state["max_iterations"]
        
        status = "✅ GOOD" if is_good else "🔄 NEEDS IMPROVEMENT"
        print(f"   {status} (Length: {len(state['content'])} chars)")
        
        return {"is_good": is_good}
    
    print_step(2, "Define routing with loop")
    
    def should_continue(state: IterativeState) -> Literal["improve", "end"]:
        """Decide whether to continue improving or finish."""
        if state["is_good"] or state["iteration"] >= state["max_iterations"]:
            return "end"
        return "improve"
    
    print("""
    Routing logic:
    - If content is good OR max iterations reached → END
    - Otherwise → loop back to improve
    """)
    
    print_step(3, "Build cyclic graph")
    
    workflow = StateGraph(IterativeState)
    workflow.add_node("improve", improve_content)
    workflow.add_node("evaluate", evaluate_content)
    
    workflow.add_edge(START, "improve")
    workflow.add_edge("improve", "evaluate")
    
    # Conditional edge that can loop back
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "improve": "improve",  # Loop back
            "end": END
        }
    )
    
    app = workflow.compile()
    
    print_step(4, "Run iterative improvement")
    
    initial_content = "AI is good."
    print(f"\n   Initial: {initial_content}")
    
    result = app.invoke({
        "content": initial_content,
        "iteration": 0,
        "max_iterations": 3,
        "is_good": False
    })
    
    print_response(
        f"Iterations: {result['iteration']}\n\n"
        f"Final content:\n{result['content']}",
        "Final Result"
    )


def example_4_parallel_and_conditional():
    """
    Example 4: Combining Parallel and Conditional
    Complex workflow with both patterns
    """
    print_section(
        "Example 4: Parallel + Conditional",
        "Combine parallel execution with conditional routing"
    )
    
    print_step(1, "Create document processing workflow")
    
    class DocState(TypedDict):
        document: str
        word_count: int
        has_urls: bool
        category: str
        processing_path: str
    
    def count_words(state: DocState) -> DocState:
        """Count words in document."""
        count = len(state["document"].split())
        print(f"   📊 Word count: {count}")
        return {"word_count": count}
    
    def check_urls(state: DocState) -> DocState:
        """Check if document contains URLs."""
        has_urls = "http" in state["document"].lower()
        print(f"   🔗 Contains URLs: {has_urls}")
        return {"has_urls": has_urls}
    
    def categorize(state: DocState) -> DocState:
        """Categorize based on length."""
        if state["word_count"] < 100:
            category = "short"
        elif state["word_count"] < 500:
            category = "medium"
        else:
            category = "long"
        
        print(f"   📁 Category: {category}")
        return {"category": category}
    
    def process_short(state: DocState) -> DocState:
        """Process short documents."""
        return {"processing_path": "Short document - quick review"}
    
    def process_medium(state: DocState) -> DocState:
        """Process medium documents."""
        return {"processing_path": "Medium document - standard review"}
    
    def process_long(state: DocState) -> DocState:
        """Process long documents."""
        return {"processing_path": "Long document - detailed review"}
    
    print_step(2, "Define routing logic")
    
    def route_by_category(state: DocState) -> Literal["short", "medium", "long"]:
        """Route based on document category."""
        return state["category"]
    
    print_step(3, "Build complex graph")
    
    workflow = StateGraph(DocState)
    
    # Analysis nodes (run in parallel)
    workflow.add_node("count_words", count_words)
    workflow.add_node("check_urls", check_urls)
    workflow.add_node("categorize", categorize)
    
    # Processing nodes
    workflow.add_node("process_short", process_short)
    workflow.add_node("process_medium", process_medium)
    workflow.add_node("process_long", process_long)
    
    # Parallel analysis
    workflow.add_edge(START, "count_words")
    workflow.add_edge(START, "check_urls")
    workflow.add_edge("count_words", "categorize")
    workflow.add_edge("check_urls", "categorize")
    
    # Conditional routing after categorization
    workflow.add_conditional_edges(
        "categorize",
        route_by_category,
        {
            "short": "process_short",
            "medium": "process_medium",
            "long": "process_long"
        }
    )
    
    # All routes end
    workflow.add_edge("process_short", END)
    workflow.add_edge("process_medium", END)
    workflow.add_edge("process_long", END)
    
    app = workflow.compile()
    
    print_step(4, "Process different documents")
    
    documents = [
        "Short text here.",
        "This is a medium length document. " * 30,
        "This is a very long document. " * 200,
    ]
    
    for i, doc in enumerate(documents, 1):
        print(f"\n   📄 Document {i} (preview): {doc[:50]}...")
        result = app.invoke({
            "document": doc,
            "word_count": 0,
            "has_urls": False,
            "category": "",
            "processing_path": ""
        })
        print(f"   ➡️  {result['processing_path']}")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" CONDITIONAL LANGGRAPH - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_simple_conditional()
        input("\nPress Enter to continue to next example...")
        
        example_2_multi_way_routing()
        input("\nPress Enter to continue to next example...")
        
        example_3_cyclic_workflow()
        input("\nPress Enter to continue to next example...")
        
        example_4_parallel_and_conditional()
        
        print("\n" + "="*70)
        print(" ✅ All conditional graph examples completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

