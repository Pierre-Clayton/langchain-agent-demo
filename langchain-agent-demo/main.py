"""
LangChain & LangGraph Interactive Demo
======================================

Main entry point for the educational demo.
Provides an interactive menu to explore all examples.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from config.settings import validate_api_keys

console = Console()


def print_welcome():
    """Display welcome message."""
    welcome_text = """
    🤖 [bold cyan]LangChain & LangGraph Educational Demo[/bold cyan]
    
    A comprehensive, hands-on demonstration of AI agent capabilities.
    
    [yellow]Features:[/yellow]
    • 🔗 Basic Chains & Prompts
    • 🛠️  Agents with Tools
    • 🧠 Memory & Conversations
    • 📊 LangGraph State Machines
    • 👥 Multi-Agent Systems
    • 📄 RAG (Retrieval Augmented Generation)
    • 🔌 MCP Integration
    • 📈 LangSmith Monitoring
    
    [dim]Navigate through examples to learn step-by-step![/dim]
    """
    
    panel = Panel(welcome_text, border_style="blue", padding=(1, 2))
    console.print(panel)


def create_menu():
    """Create the main menu."""
    table = Table(title="📚 Example Categories", show_header=True, header_style="bold magenta")
    table.add_column("No.", style="cyan", width=4)
    table.add_column("Category", style="green", width=30)
    table.add_column("Description", style="white", width=50)
    
    examples = [
        ("1", "Basic Chains", "Learn fundamental LangChain concepts"),
        ("2", "Prompt Engineering", "Master prompt templates and patterns"),
        ("3", "LLM Interactions", "Explore different LLM usage patterns"),
        ("4", "Simple Agents", "Agents with basic tools"),
        ("5", "ReAct Agents", "Advanced reasoning agents"),
        ("6", "Conversation Memory", "Add memory to agents and chains"),
        ("7", "LangGraph Basics", "Introduction to state machines"),
        ("8", "Conditional Graphs", "Dynamic routing and branching"),
        ("9", "Multi-Agent Systems", "Coordinated agent teams"),
        ("10", "RAG System", "Retrieval Augmented Generation"),
        ("11", "MCP Integration", "Model Context Protocol examples"),
        ("12", "LangSmith Monitoring", "Observability and debugging"),
        ("13", "Run All Examples", "Execute all examples sequentially"),
        ("0", "Exit", "Quit the demo"),
    ]
    
    for no, category, description in examples:
        table.add_row(no, category, description)
    
    return table


def run_example(choice: str):
    """
    Run the selected example.
    
    Args:
        choice: User's menu choice
    """
    examples = {
        "1": ("src.01_basics.chains", "Basic Chains"),
        "2": ("src.01_basics.prompts", "Prompt Engineering"),
        "3": ("src.01_basics.llm_examples", "LLM Interactions"),
        "4": ("src.02_agents.simple_agent", "Simple Agents"),
        "5": ("src.02_agents.react_agent", "ReAct Agents"),
        "6": ("src.03_memory.conversation_memory", "Conversation Memory"),
        "7": ("src.04_langgraph.simple_graph", "LangGraph Basics"),
        "8": ("src.04_langgraph.conditional_graph", "Conditional Graphs"),
        "9": ("src.05_multi_agent.research_team", "Multi-Agent Systems"),
        "10": ("src.06_rag.qa_system", "RAG System"),
        "11": ("src.07_mcp.mcp_integration", "MCP Integration"),
        "12": ("src.08_monitoring.langsmith_monitoring", "LangSmith Monitoring"),
    }
    
    if choice not in examples:
        console.print("[red]Invalid choice. Please try again.[/red]")
        return
    
    module_path, title = examples[choice]
    
    console.print(f"\n[bold green]🚀 Launching: {title}[/bold green]\n")
    console.print("[dim]Press Ctrl+C to return to menu[/dim]\n")
    
    try:
        # Import and run the module
        module = __import__(module_path, fromlist=["main"])
        module.main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⏸️  Example interrupted. Returning to menu...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error running example: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    console.print("\n" + "="*70)
    input("\nPress Enter to return to main menu...")


def run_all_examples():
    """Run all examples sequentially."""
    console.print("\n[bold cyan]🎯 Running All Examples[/bold cyan]\n")
    console.print("[yellow]This will take a while. Press Ctrl+C to stop.[/yellow]\n")
    
    examples = [
        ("src.01_basics.chains", "1. Basic Chains"),
        ("src.01_basics.prompts", "2. Prompt Engineering"),
        ("src.01_basics.llm_examples", "3. LLM Interactions"),
        ("src.02_agents.simple_agent", "4. Simple Agents"),
        ("src.02_agents.react_agent", "5. ReAct Agents"),
        ("src.03_memory.conversation_memory", "6. Conversation Memory"),
        ("src.04_langgraph.simple_graph", "7. LangGraph Basics"),
        ("src.04_langgraph.conditional_graph", "8. Conditional Graphs"),
        ("src.05_multi_agent.research_team", "9. Multi-Agent Systems"),
        ("src.06_rag.qa_system", "10. RAG System"),
        ("src.07_mcp.mcp_integration", "11. MCP Integration"),
        ("src.08_monitoring.langsmith_monitoring", "12. LangSmith Monitoring"),
    ]
    
    for i, (module_path, title) in enumerate(examples, 1):
        console.print(f"\n[bold green]{'='*70}[/bold green]")
        console.print(f"[bold green]{title}[/bold green]")
        console.print(f"[bold green]{'='*70}[/bold green]\n")
        
        try:
            module = __import__(module_path, fromlist=["main"])
            module.main()
        except KeyboardInterrupt:
            console.print("\n\n[yellow]⏸️  Interrupted. Returning to menu...[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Error in {title}: {e}[/red]")
            response = Prompt.ask(
                "\n[yellow]Continue with next example?[/yellow]",
                choices=["y", "n"],
                default="y"
            )
            if response.lower() != "y":
                break
        
        if i < len(examples):
            input("\n[dim]Press Enter to continue to next example...[/dim]")
    
    console.print("\n[bold green]✅ All examples completed![/bold green]")
    input("\nPress Enter to return to main menu...")


def show_quick_start():
    """Show quick start guide."""
    guide = """
    [bold cyan]🚀 Quick Start Guide[/bold cyan]
    
    [yellow]1. Setup (if not done)[/yellow]
       • Copy .env.example to .env
       • Add your OPENAI_API_KEY
       • Run: pip install -r requirements.txt
    
    [yellow]2. Choose Your Path[/yellow]
       • [green]Beginner?[/green] Start with: 1 → 2 → 4 → 7
       • [green]Intermediate?[/green] Jump to: 5 → 6 → 8 → 10
       • [green]Advanced?[/green] Explore: 9 (Multi-Agent) & 10 (RAG)
    
    [yellow]3. Learning Tips[/yellow]
       • Each example is self-contained
       • Read the code comments - they're educational!
       • Experiment with the parameters
       • Try modifying the examples
    
    [yellow]4. Need Help?[/yellow]
       • Check README.md for detailed docs
       • Each module has extensive docstrings
       • Examples include explanations
    
    [dim]Happy Learning! 🎓[/dim]
    """
    
    panel = Panel(guide, border_style="cyan", padding=(1, 2))
    console.print(panel)
    input("\nPress Enter to continue...")


def show_about():
    """Show about information."""
    about = """
    [bold cyan]About This Demo[/bold cyan]
    
    [yellow]Purpose:[/yellow]
    This is a comprehensive educational resource for learning LangChain
    and LangGraph. It covers everything from basic concepts to advanced
    multi-agent systems.
    
    [yellow]What You'll Learn:[/yellow]
    • How to build AI agents from scratch
    • Different types of chains and their uses
    • Memory management in conversations
    • State machine workflows with LangGraph
    • Multi-agent coordination patterns
    • RAG for knowledge-based Q&A
    
    [yellow]Technologies:[/yellow]
    • LangChain 0.2.x
    • LangGraph 0.2.x
    • OpenAI GPT models
    • FAISS vector store
    • Python 3.10+
    
    [yellow]Author:[/yellow]
    Created as an educational resource for the AI/ML community
    
    [yellow]License:[/yellow]
    MIT License - Free to use and modify
    
    [dim]Version 1.0.0[/dim]
    """
    
    panel = Panel(about, border_style="blue", padding=(1, 2))
    console.print(panel)
    input("\nPress Enter to continue...")


def main():
    """Main application loop."""
    # Clear screen (cross-platform)
    console.clear()
    
    # Show welcome message
    print_welcome()
    
    # Validate API keys
    if not validate_api_keys():
        console.print("\n[red]⚠️  API keys not configured![/red]\n")
        console.print("Please set up your .env file with required API keys.")
        console.print("See .env.example for template.\n")
        response = Prompt.ask(
            "Continue anyway? (Some examples may not work)",
            choices=["y", "n"],
            default="n"
        )
        if response.lower() != "y":
            console.print("\n[yellow]👋 See you later![/yellow]")
            return
    
    # Main menu loop
    while True:
        console.print("\n")
        menu = create_menu()
        console.print(menu)
        console.print("\n")
        
        # Show additional options
        console.print("[dim]Additional options:[/dim]")
        console.print("[dim]  • Type 'help' for quick start guide[/dim]")
        console.print("[dim]  • Type 'about' for more information[/dim]\n")
        
        choice = Prompt.ask(
            "[bold cyan]Select an option[/bold cyan]",
            default="1"
        )
        
        # Handle special commands
        if choice.lower() == "help":
            show_quick_start()
            continue
        
        if choice.lower() == "about":
            show_about()
            continue
        
        if choice == "0":
            console.print("\n[bold green]👋 Thanks for exploring LangChain & LangGraph![/bold green]")
            console.print("[dim]Keep building amazing AI applications![/dim]\n")
            break
        
        if choice == "13":
            run_all_examples()
            continue
        
        # Run selected example
        run_example(choice)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {e}[/red]")
        import traceback
        traceback.print_exc()

