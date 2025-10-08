"""
Run All Examples Script
=======================

Execute all examples sequentially without interactive prompts.
Useful for testing or demonstrations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import validate_api_keys


def main():
    """Run all examples in order."""
    if not validate_api_keys():
        print("❌ API keys not configured. Please set up your .env file.")
        return
    
    examples = [
        ("src.01_basics.chains", "Basic Chains"),
        ("src.01_basics.prompts", "Prompt Engineering"),
        ("src.01_basics.llm_examples", "LLM Interactions"),
        ("src.02_agents.simple_agent", "Simple Agents"),
        ("src.02_agents.react_agent", "ReAct Agents"),
        ("src.03_memory.conversation_memory", "Conversation Memory"),
        ("src.04_langgraph.simple_graph", "LangGraph Basics"),
        ("src.04_langgraph.conditional_graph", "Conditional Graphs"),
        ("src.05_multi_agent.research_team", "Multi-Agent Systems"),
        ("src.06_rag.qa_system", "RAG System"),
        ("src.07_mcp.mcp_integration", "MCP Integration"),
        ("src.08_monitoring.langsmith_monitoring", "LangSmith Monitoring"),
    ]
    
    print("\n" + "="*70)
    print("  RUNNING ALL EXAMPLES".center(70))
    print("="*70 + "\n")
    
    for i, (module_path, title) in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(examples)}] {title}".center(70))
        print(f"{'='*70}\n")
        
        try:
            module = __import__(module_path, fromlist=["main"])
            module.main()
        except KeyboardInterrupt:
            print("\n\n⏸️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in {title}: {e}")
            import traceback
            traceback.print_exc()
            response = input("\nContinue with next example? (y/n): ")
            if response.lower() != "y":
                break
    
    print("\n" + "="*70)
    print("  ✅ ALL EXAMPLES COMPLETED".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

