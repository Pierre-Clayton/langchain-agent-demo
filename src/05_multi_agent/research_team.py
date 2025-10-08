"""
Multi-Agent Research Team
=========================

This module demonstrates a complete multi-agent system:
- Multiple specialized agents
- Agent coordination
- Task delegation
- Collaborative problem solving

Learning Objectives:
- Build specialized agents
- Coordinate multiple agents
- Implement supervisor patterns
- Create agent workflows
"""

import sys
from pathlib import Path
from typing import TypedDict, Literal, Annotated
from operator import add

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from src.utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


def example_1_specialized_agents():
    """
    Example 1: Team of Specialized Agents
    Create agents with different roles
    """
    print_section(
        "Example 1: Specialized Agents",
        "Build a research team with different specializations"
    )
    
    print_step(1, "Define the team state")
    
    class TeamState(TypedDict):
        task: str
        research_notes: str
        analysis: str
        summary: str
        final_report: str
    
    print_step(2, "Create specialized agents")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    
    def researcher_agent(state: TeamState) -> TeamState:
        """Research Agent: Gather information."""
        print("\n   🔍 RESEARCHER: Gathering information...")
        
        prompt = ChatPromptTemplate.from_template(
            "You are a researcher. Gather key information about: {task}\n"
            "Provide 3-4 key points with facts."
        )
        
        chain = prompt | llm
        research = chain.invoke({"task": state["task"]}).content
        
        print(f"   ✅ Research complete (preview): {research[:100]}...")
        return {"research_notes": research}
    
    def analyst_agent(state: TeamState) -> TeamState:
        """Analyst Agent: Analyze information."""
        print("\n   📊 ANALYST: Analyzing research...")
        
        prompt = ChatPromptTemplate.from_template(
            "You are an analyst. Analyze these research notes:\n\n{research}\n\n"
            "Provide insights, patterns, and implications."
        )
        
        chain = prompt | llm
        analysis = chain.invoke({"research": state["research_notes"]}).content
        
        print(f"   ✅ Analysis complete (preview): {analysis[:100]}...")
        return {"analysis": analysis}
    
    def writer_agent(state: TeamState) -> TeamState:
        """Writer Agent: Create final report."""
        print("\n   ✍️  WRITER: Creating final report...")
        
        prompt = ChatPromptTemplate.from_template(
            "You are a technical writer. Create a well-structured report.\n\n"
            "Research:\n{research}\n\n"
            "Analysis:\n{analysis}\n\n"
            "Write a clear, engaging report."
        )
        
        chain = prompt | llm
        report = chain.invoke({
            "research": state["research_notes"],
            "analysis": state["analysis"]
        }).content
        
        print(f"   ✅ Report complete")
        return {"final_report": report}
    
    print_step(3, "Build the team workflow")
    
    workflow = StateGraph(TeamState)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("writer", writer_agent)
    
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", END)
    
    app = workflow.compile()
    
    print_step(4, "Execute team workflow")
    
    task = "The impact of quantum computing on cryptography"
    print(f"\n   📋 Task: {task}\n")
    
    result = app.invoke({
        "task": task,
        "research_notes": "",
        "analysis": "",
        "summary": "",
        "final_report": ""
    })
    
    print_response(result["final_report"], "Team Report")


def example_2_supervisor_pattern():
    """
    Example 2: Supervisor Agent Pattern
    One agent coordinates others
    """
    print_section(
        "Example 2: Supervisor Pattern",
        "A supervisor agent delegates tasks to worker agents"
    )
    
    print_step(1, "Define supervisor state")
    
    class SupervisorState(TypedDict):
        task: str
        next_agent: str
        messages: Annotated[list[str], add]
        final_answer: str
    
    print_step(2, "Create worker agents")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.3)
    
    def web_researcher(state: SupervisorState) -> SupervisorState:
        """Simulated web researcher."""
        print("   🌐 Web Researcher working...")
        
        prompt = f"Research online about: {state['task']}\nProvide 2-3 key findings."
        result = llm.invoke(prompt).content
        
        return {
            "messages": [f"Web Research: {result}"],
            "next_agent": "supervisor"
        }
    
    def data_analyst(state: SupervisorState) -> SupervisorState:
        """Data analysis agent."""
        print("   📊 Data Analyst working...")
        
        prompt = f"Analyze data related to: {state['task']}\nProvide insights."
        result = llm.invoke(prompt).content
        
        return {
            "messages": [f"Data Analysis: {result}"],
            "next_agent": "supervisor"
        }
    
    def report_writer(state: SupervisorState) -> SupervisorState:
        """Report writing agent."""
        print("   📝 Report Writer working...")
        
        all_info = "\n\n".join(state["messages"])
        prompt = f"Based on this information:\n\n{all_info}\n\nWrite a brief summary."
        result = llm.invoke(prompt).content
        
        return {
            "messages": [f"Report: {result}"],
            "final_answer": result,
            "next_agent": "end"
        }
    
    def supervisor(state: SupervisorState) -> SupervisorState:
        """Supervisor decides next step."""
        print("   👔 Supervisor deciding next step...")
        
        # Simple rule-based supervision
        if len(state["messages"]) == 0:
            next_agent = "researcher"
        elif len(state["messages"]) == 1:
            next_agent = "analyst"
        elif len(state["messages"]) == 2:
            next_agent = "writer"
        else:
            next_agent = "end"
        
        print(f"      → Delegating to: {next_agent}")
        return {"next_agent": next_agent}
    
    print_step(3, "Build supervised workflow")
    
    def route_supervisor(state: SupervisorState) -> Literal["researcher", "analyst", "writer", "end"]:
        """Route based on supervisor decision."""
        return state["next_agent"]
    
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", web_researcher)
    workflow.add_node("analyst", data_analyst)
    workflow.add_node("writer", report_writer)
    
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "end": END
        }
    )
    
    # All workers report back to supervisor
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("writer", END)
    
    app = workflow.compile()
    
    print_step(4, "Execute supervised workflow")
    
    task = "Electric vehicle adoption trends"
    print(f"\n   📋 Task: {task}\n")
    
    result = app.invoke({
        "task": task,
        "next_agent": "",
        "messages": [],
        "final_answer": ""
    })
    
    print_response(result["final_answer"], "Final Report")


def example_3_collaborative_agents():
    """
    Example 3: Collaborative Agents
    Agents working together and reviewing each other
    """
    print_section(
        "Example 3: Collaborative Agents",
        "Agents collaborate and provide feedback to each other"
    )
    
    print_step(1, "Define collaboration state")
    
    class CollabState(TypedDict):
        topic: str
        draft: str
        feedback: str
        revision: str
        final: str
        iteration: int
    
    print_step(2, "Create collaborative agents")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    
    def writer(state: CollabState) -> CollabState:
        """Writer creates content."""
        print("\n   ✍️  Writer: Creating content...")
        
        if state["iteration"] == 0:
            prompt = f"Write a short article (4-5 sentences) about: {state['topic']}"
            draft = llm.invoke(prompt).content
        else:
            prompt = f"Revise this article based on feedback:\n\nArticle: {state['draft']}\n\nFeedback: {state['feedback']}"
            draft = llm.invoke(prompt).content
        
        return {"draft": draft}
    
    def critic(state: CollabState) -> CollabState:
        """Critic provides feedback."""
        print("\n   🎭 Critic: Reviewing content...")
        
        prompt = f"""Review this article:\n\n{state['draft']}\n\n
        Provide constructive feedback on:
        1. Clarity
        2. Engagement
        3. Structure
        
        Be specific but concise."""
        
        feedback = llm.invoke(prompt).content
        print(f"   📝 Feedback provided")
        
        return {"feedback": feedback, "iteration": state["iteration"] + 1}
    
    def finalizer(state: CollabState) -> CollabState:
        """Finalize the content."""
        print("\n   ✅ Finalizing content...")
        return {"final": state["draft"]}
    
    print_step(3, "Build collaborative workflow")
    
    def should_revise(state: CollabState) -> Literal["writer", "finalizer"]:
        """Decide if revision is needed."""
        if state["iteration"] < 2:  # Allow up to 2 revisions
            return "writer"
        return "finalizer"
    
    workflow = StateGraph(CollabState)
    workflow.add_node("writer", writer)
    workflow.add_node("critic", critic)
    workflow.add_node("finalizer", finalizer)
    
    workflow.add_edge(START, "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "writer": "writer",
            "finalizer": "finalizer"
        }
    )
    workflow.add_edge("finalizer", END)
    
    app = workflow.compile()
    
    print_step(4, "Run collaborative process")
    
    topic = "The benefits of meditation for developers"
    print(f"\n   📋 Topic: {topic}\n")
    
    result = app.invoke({
        "topic": topic,
        "draft": "",
        "feedback": "",
        "revision": "",
        "final": "",
        "iteration": 0
    })
    
    print(f"\n   🔄 Total iterations: {result['iteration']}")
    print_response(result["final"], "Final Article")


def example_4_parallel_agents():
    """
    Example 4: Parallel Agent Execution
    Multiple agents working simultaneously
    """
    print_section(
        "Example 4: Parallel Agents",
        "Multiple agents working on different aspects simultaneously"
    )
    
    print_step(1, "Define parallel state")
    
    class ParallelState(TypedDict):
        query: str
        perspectives: Annotated[list[str], add]
        synthesis: str
    
    print_step(2, "Create perspective agents")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0.7)
    
    def technical_perspective(state: ParallelState) -> ParallelState:
        """Technical viewpoint."""
        print("   💻 Technical Expert analyzing...")
        
        prompt = f"From a technical perspective, analyze: {state['query']}"
        result = llm.invoke(prompt).content
        
        return {"perspectives": [f"**Technical View:**\n{result}"]}
    
    def business_perspective(state: ParallelState) -> ParallelState:
        """Business viewpoint."""
        print("   💼 Business Analyst analyzing...")
        
        prompt = f"From a business perspective, analyze: {state['query']}"
        result = llm.invoke(prompt).content
        
        return {"perspectives": [f"**Business View:**\n{result}"]}
    
    def user_perspective(state: ParallelState) -> ParallelState:
        """User viewpoint."""
        print("   👤 UX Expert analyzing...")
        
        prompt = f"From a user experience perspective, analyze: {state['query']}"
        result = llm.invoke(prompt).content
        
        return {"perspectives": [f"**User View:**\n{result}"]}
    
    def synthesize(state: ParallelState) -> ParallelState:
        """Synthesize all perspectives."""
        print("\n   🔄 Synthesizing all perspectives...")
        
        all_perspectives = "\n\n".join(state["perspectives"])
        prompt = f"Synthesize these perspectives into a balanced conclusion:\n\n{all_perspectives}"
        synthesis = llm.invoke(prompt).content
        
        return {"synthesis": synthesis}
    
    print_step(3, "Build parallel workflow")
    
    workflow = StateGraph(ParallelState)
    workflow.add_node("technical", technical_perspective)
    workflow.add_node("business", business_perspective)
    workflow.add_node("user", user_perspective)
    workflow.add_node("synthesize", synthesize)
    
    # All perspectives run in parallel
    workflow.add_edge(START, "technical")
    workflow.add_edge(START, "business")
    workflow.add_edge(START, "user")
    
    # All converge at synthesis
    workflow.add_edge("technical", "synthesize")
    workflow.add_edge("business", "synthesize")
    workflow.add_edge("user", "synthesize")
    
    workflow.add_edge("synthesize", END)
    
    app = workflow.compile()
    
    print_step(4, "Execute parallel analysis")
    
    query = "Should we adopt microservices architecture?"
    print(f"\n   ❓ Query: {query}\n")
    
    result = app.invoke({
        "query": query,
        "perspectives": [],
        "synthesis": ""
    })
    
    print("\n   📊 Individual Perspectives:")
    for i, perspective in enumerate(result["perspectives"], 1):
        print(f"\n   {i}. {perspective[:100]}...")
    
    print_response(result["synthesis"], "Synthesized Conclusion")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" MULTI-AGENT SYSTEMS - EDUCATIONAL DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_specialized_agents()
        input("\nPress Enter to continue to next example...")
        
        example_2_supervisor_pattern()
        input("\nPress Enter to continue to next example...")
        
        example_3_collaborative_agents()
        input("\nPress Enter to continue to next example...")
        
        example_4_parallel_agents()
        
        print("\n" + "="*70)
        print(" ✅ All multi-agent examples completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

