"""
Simple LangGraph Examples
=========================

This module introduces LangGraph basics:
- What is a state graph?
- Nodes and edges
- State management
- Linear workflows

Learning Objectives:
- Understand state graphs
- Create nodes (functions)
- Connect nodes with edges
- Manage state between nodes
"""

import sys
from pathlib import Path
from typing import TypedDict, Annotated
from operator import add

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from src.utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


def example_1_basic_graph():
    """
    Example 1: Basic Linear Graph
    Three nodes in a sequence
    """
    print_section(
        "Example 1: Basic Linear Graph",
        "A simple workflow with three sequential steps"
    )
    
    print_step(1, "Define the state")
    
    # State = data passed between nodes
    class State(TypedDict):
        message: str
        count: int
    
    print("""
    State structure:
    - message: str (text to process)
    - count: int (tracking progress)
    """)
    
    print_step(2, "Define node functions")
    
    def node_1(state: State) -> State:
        """First processing step."""
        print(f"   🔵 Node 1: Received '{state['message']}'")
        return {
            "message": state["message"].upper(),
            "count": state["count"] + 1
        }
    
    def node_2(state: State) -> State:
        """Second processing step."""
        print(f"   🟢 Node 2: Received '{state['message']}'")
        return {
            "message": state["message"] + " - PROCESSED",
            "count": state["count"] + 1
        }
    
    def node_3(state: State) -> State:
        """Third processing step."""
        print(f"   🟡 Node 3: Received '{state['message']}'")
        return {
            "message": f"Final: {state['message']}",
            "count": state["count"] + 1
        }
    
    print_step(3, "Build the graph")
    
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("step_1", node_1)
    workflow.add_node("step_2", node_2)
    workflow.add_node("step_3", node_3)
    
    # Add edges (connections)
    workflow.add_edge(START, "step_1")
    workflow.add_edge("step_1", "step_2")
    workflow.add_edge("step_2", "step_3")
    workflow.add_edge("step_3", END)
    
    print("""
    Graph structure:
    START → step_1 → step_2 → step_3 → END
    """)
    
    print_step(4, "Compile and run the graph")
    
    app = workflow.compile()
    
    # Execute
    result = app.invoke({"message": "hello world", "count": 0})
    
    print_response(
        f"Final message: {result['message']}\n"
        f"Steps executed: {result['count']}",
        "Result"
    )


def example_2_llm_nodes():
    """
    Example 2: Graph with LLM Nodes
    Use LLMs in graph nodes
    """
    print_section(
        "Example 2: LLM in Graph Nodes",
        "Create a content generation pipeline with LLMs"
    )
    
    print_step(1, "Define state for content creation")
    
    class ContentState(TypedDict):
        topic: str
        outline: str
        draft: str
        final: str
    
    print_step(2, "Create LLM-powered nodes")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    
    def create_outline(state: ContentState) -> ContentState:
        """Generate an outline."""
        print("   📝 Creating outline...")
        prompt = f"Create a brief outline for a blog post about: {state['topic']}"
        outline = llm.invoke(prompt).content
        return {"outline": outline}
    
    def write_draft(state: ContentState) -> ContentState:
        """Write a draft."""
        print("   ✍️  Writing draft...")
        prompt = f"Based on this outline:\n{state['outline']}\n\nWrite a short draft (3-4 sentences)."
        draft = llm.invoke(prompt).content
        return {"draft": draft}
    
    def finalize(state: ContentState) -> ContentState:
        """Finalize the content."""
        print("   ✅ Finalizing...")
        prompt = f"Polish this draft:\n{state['draft']}\n\nMake it more engaging."
        final = llm.invoke(prompt).content
        return {"final": final}
    
    print_step(3, "Build content creation pipeline")
    
    workflow = StateGraph(ContentState)
    workflow.add_node("create_outline", create_outline)
    workflow.add_node("write_draft", write_draft)
    workflow.add_node("finalize_content", finalize)
    
    workflow.add_edge(START, "create_outline")
    workflow.add_edge("create_outline", "write_draft")
    workflow.add_edge("write_draft", "finalize_content")
    workflow.add_edge("finalize_content", END)
    
    app = workflow.compile()
    
    print_step(4, "Generate content")
    
    result = app.invoke({"topic": "The benefits of meditation"})
    
    print("\n   📋 Outline:")
    print(f"   {result['outline'][:200]}...\n")
    
    print("   📄 Draft:")
    print(f"   {result['draft'][:200]}...\n")
    
    print_response(result['final'], "Final Content")


def example_3_state_accumulation():
    """
    Example 3: State Accumulation
    Accumulate values across nodes
    """
    print_section(
        "Example 3: State Accumulation",
        "Build up state values through the pipeline"
    )
    
    print_step(1, "Define state with accumulation")
    
    class AccumulateState(TypedDict):
        input: str
        results: Annotated[list[str], add]  # Accumulate results
    
    print("   Using Annotated with 'add' to accumulate results")
    
    print_step(2, "Create processing nodes")
    
    def analyze_length(state: AccumulateState) -> AccumulateState:
        """Analyze text length."""
        length = len(state["input"])
        return {"results": [f"Length: {length} characters"]}
    
    def count_words(state: AccumulateState) -> AccumulateState:
        """Count words."""
        words = len(state["input"].split())
        return {"results": [f"Words: {words}"]}
    
    def check_uppercase(state: AccumulateState) -> AccumulateState:
        """Check uppercase letters."""
        uppercase = sum(1 for c in state["input"] if c.isupper())
        return {"results": [f"Uppercase letters: {uppercase}"]}
    
    print_step(3, "Build parallel analysis graph")
    
    workflow = StateGraph(AccumulateState)
    workflow.add_node("length", analyze_length)
    workflow.add_node("words", count_words)
    workflow.add_node("uppercase", check_uppercase)
    
    # All nodes run in parallel from START
    workflow.add_edge(START, "length")
    workflow.add_edge(START, "words")
    workflow.add_edge(START, "uppercase")
    
    # All converge to END
    workflow.add_edge("length", END)
    workflow.add_edge("words", END)
    workflow.add_edge("uppercase", END)
    
    app = workflow.compile()
    
    print_step(4, "Analyze text")
    
    text = "LangChain and LangGraph are Powerful Tools for AI!"
    result = app.invoke({"input": text, "results": []})
    
    print(f"\n   Input: {text}\n")
    print("   📊 Analysis Results:")
    for r in result["results"]:
        print(f"      • {r}")


def example_4_streaming_output():
    """
    Example 4: Streaming Graph Execution
    See intermediate results as they happen
    """
    print_section(
        "Example 4: Streaming Execution",
        "Watch the graph execute step by step"
    )
    
    print_step(1, "Create a multi-step workflow")
    
    class StreamState(TypedDict):
        number: int
        operations: list[str]
    
    def multiply_by_2(state: StreamState) -> StreamState:
        """Multiply by 2."""
        new_num = state["number"] * 2
        return {
            "number": new_num,
            "operations": [f"Multiplied by 2: {state['number']} → {new_num}"]
        }
    
    def add_10(state: StreamState) -> StreamState:
        """Add 10."""
        new_num = state["number"] + 10
        return {
            "number": new_num,
            "operations": [f"Added 10: {state['number']} → {new_num}"]
        }
    
    def square(state: StreamState) -> StreamState:
        """Square the number."""
        new_num = state["number"] ** 2
        return {
            "number": new_num,
            "operations": [f"Squared: {state['number']} → {new_num}"]
        }
    
    workflow = StateGraph(StreamState)
    workflow.add_node("multiply", multiply_by_2)
    workflow.add_node("add", add_10)
    workflow.add_node("square", square)
    
    workflow.add_edge(START, "multiply")
    workflow.add_edge("multiply", "add")
    workflow.add_edge("add", "square")
    workflow.add_edge("square", END)
    
    app = workflow.compile()
    
    print_step(2, "Stream the execution")
    
    initial_value = 5
    print(f"\n   Starting value: {initial_value}\n")
    
    for step in app.stream({"number": initial_value, "operations": []}):
        node_name = list(step.keys())[0]
        state = step[node_name]
        if "operations" in state and state["operations"]:
            print(f"   🔄 {node_name}: {state['operations'][0]}")
    
    print(f"\n   🎯 Final value: {state['number']}")


def example_5_graph_visualization():
    """
    Example 5: Understanding Graph Structure
    Visualize how graphs are organized
    """
    print_section(
        "Example 5: Graph Visualization",
        "Understanding graph structure and flow"
    )
    
    print_step(1, "Create a complex graph")
    
    class State(TypedDict):
        value: str
        paths: Annotated[list[str], add]  # Track all paths taken
    
    workflow = StateGraph(State)
    
    # Add multiple nodes
    workflow.add_node("A", lambda s: {"value": s["value"] + "→A", "paths": ["A"]})
    workflow.add_node("B", lambda s: {"paths": ["B"]})
    workflow.add_node("C", lambda s: {"paths": ["C"]})
    workflow.add_node("D", lambda s: {"paths": ["D"]})
    
    # Create a diamond pattern
    workflow.add_edge(START, "A")
    workflow.add_edge("A", "B")
    workflow.add_edge("A", "C")
    workflow.add_edge("B", "D")
    workflow.add_edge("C", "D")
    workflow.add_edge("D", END)
    
    print("""
    Graph structure:
    
           START
             |
             A
            / \\
           B   C
            \\ /
             D
             |
            END
    
    Node A branches to B and C (parallel)
    Both B and C converge at D
    """)
    
    app = workflow.compile()
    
    print_step(2, "Execute the graph")
    
    result = app.invoke({"value": "START", "paths": []})
    print_response(
        f"Final value: {result['value']}\n"
        f"Nodes executed: {' → '.join(result['paths'])}",
        "Execution Flow"
    )


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" LANGGRAPH BASICS - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_basic_graph()
        input("\nPress Enter to continue to next example...")
        
        example_2_llm_nodes()
        input("\nPress Enter to continue to next example...")
        
        example_3_state_accumulation()
        input("\nPress Enter to continue to next example...")
        
        example_4_streaming_output()
        input("\nPress Enter to continue to next example...")
        
        example_5_graph_visualization()
        
        print("\n" + "="*70)
        print(" ✅ All LangGraph basics completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

