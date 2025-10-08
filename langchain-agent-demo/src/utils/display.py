"""
Display utilities for pretty printing demo outputs.
Uses the rich library for beautiful terminal output.
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from typing import Any, Optional

console = Console()


def print_section(title: str, description: Optional[str] = None):
    """
    Print a section header.
    
    Args:
        title: Section title
        description: Optional description
    """
    console.print()
    console.rule(f"[bold blue]{title}[/bold blue]")
    if description:
        console.print(f"[dim]{description}[/dim]")
    console.print()


def print_response(content: str, title: str = "Response", style: str = "green"):
    """
    Print a response in a panel.
    
    Args:
        content: Response content
        title: Panel title
        style: Color style
    """
    panel = Panel(
        content,
        title=f"[bold]{title}[/bold]",
        border_style=style,
        padding=(1, 2)
    )
    console.print(panel)


def print_step(step_num: int, description: str, details: Optional[str] = None):
    """
    Print a step in a process.
    
    Args:
        step_num: Step number
        description: Step description
        details: Optional details
    """
    console.print(f"\n[bold cyan]Step {step_num}:[/bold cyan] {description}")
    if details:
        console.print(f"[dim]{details}[/dim]")


def print_error(message: str):
    """
    Print an error message.
    
    Args:
        message: Error message
    """
    console.print(f"\n[bold red]❌ Error:[/bold red] {message}\n")


def print_success(message: str):
    """
    Print a success message.
    
    Args:
        message: Success message
    """
    console.print(f"\n[bold green]✅ {message}[/bold green]\n")


def print_code(code: str, language: str = "python"):
    """
    Print syntax-highlighted code.
    
    Args:
        code: Code to display
        language: Programming language
    """
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)


def print_dict(data: dict, title: str = "Data"):
    """
    Print a dictionary in a formatted table.
    
    Args:
        data: Dictionary to display
        title: Table title
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in data.items():
        table.add_row(str(key), str(value))
    
    console.print(table)


def print_markdown(text: str):
    """
    Print markdown-formatted text.
    
    Args:
        text: Markdown text
    """
    md = Markdown(text)
    console.print(md)


def print_agent_thought(thought: str):
    """
    Print an agent's thought process.
    
    Args:
        thought: Agent thought
    """
    console.print(f"[yellow]💭 Thought:[/yellow] {thought}")


def print_agent_action(action: str, action_input: str):
    """
    Print an agent's action.
    
    Args:
        action: Action name
        action_input: Action input
    """
    console.print(f"[blue]🔧 Action:[/blue] {action}")
    console.print(f"[dim]   Input: {action_input}[/dim]")


def print_agent_observation(observation: str):
    """
    Print an agent's observation.
    
    Args:
        observation: Observation result
    """
    # Truncate long observations
    if len(observation) > 200:
        observation = observation[:200] + "..."
    console.print(f"[green]👁️  Observation:[/green] {observation}")

