"""
Setup Script
===========

Quick setup script for the LangChain Agent Demo.
"""

import os
import sys
from pathlib import Path
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required.")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example not found")
        return False
    
    # Copy template
    with open(env_example, "r") as f:
        content = f.read()
    
    with open(env_file, "w") as f:
        f.write(content)
    
    print("✅ Created .env file from template")
    print("⚠️  Please edit .env and add your API keys!")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "examples/sample_documents",
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")
    return True


def main():
    """Run setup."""
    print("="*70)
    print("  LangChain Agent Demo - Setup".center(70))
    print("="*70 + "\n")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Create .env file
    if not create_env_file():
        return False
    
    # Ask about installing dependencies
    print("\n" + "-"*70)
    response = input("\n📦 Install Python dependencies now? (y/n): ").lower()
    
    if response == "y":
        if not install_dependencies():
            return False
    else:
        print("\n⚠️  Remember to install dependencies later with:")
        print("   pip install -r requirements.txt")
    
    # Success message
    print("\n" + "="*70)
    print("  ✅ Setup Complete!".center(70))
    print("="*70 + "\n")
    
    print("📝 Next Steps:\n")
    print("1. Edit .env file and add your OPENAI_API_KEY")
    print("   (Get one from: https://platform.openai.com/api-keys)")
    print()
    print("2. Run the interactive demo:")
    print("   python main.py")
    print()
    print("3. Or run individual examples:")
    print("   python -m src.01_basics.chains")
    print()
    print("4. Check README.md for detailed documentation")
    print()
    print("Happy learning! 🚀\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏸️  Setup interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

