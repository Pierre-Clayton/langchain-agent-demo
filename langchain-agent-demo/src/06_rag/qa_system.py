"""
RAG Question-Answering System
=============================

This module demonstrates RAG (Retrieval Augmented Generation):
- Document loading and chunking
- Vector embeddings
- Semantic search
- Question answering with context

Learning Objectives:
- Build a complete RAG pipeline
- Understand vector embeddings
- Implement semantic search
- Generate context-aware answers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document
from src.utils.display import print_section, print_response, print_step
from config.settings import settings, validate_api_keys


# Sample documents about LangChain
SAMPLE_DOCS = [
    """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are context-aware and can reason about their responses.
    LangChain provides standard interfaces for chains, lots of integrations with other tools,
    and end-to-end chains for common applications.
    """,
    """
    LangGraph is a library built on top of LangChain that adds support for creating
    stateful, multi-actor applications with LLMs. It allows you to define workflows
    as graphs with nodes and edges, making complex AI systems easier to build and visualize.
    LangGraph is particularly useful for multi-agent systems and complex decision trees.
    """,
    """
    Chains in LangChain are sequences of calls to components. The most basic chain is
    a simple sequence of prompt → LLM → output parser. More complex chains can include
    multiple LLMs, different prompts, and various processing steps. Chains can be
    composed together to create sophisticated applications.
    """,
    """
    Agents in LangChain use LLMs to determine which actions to take and in what order.
    They combine reasoning with tool usage. An agent has access to a suite of tools
    and based on the user input, decides which tools to call and what to pass to them.
    Agents follow the ReAct pattern: Reasoning and Acting in a loop.
    """,
    """
    Memory in LangChain allows chatbots and agents to remember previous interactions.
    Different types of memory include ConversationBufferMemory (stores all messages),
    ConversationWindowMemory (stores recent messages), and ConversationSummaryMemory
    (creates summaries of conversations). Memory can be added to chains and agents.
    """,
    """
    Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses
    by providing relevant context from a knowledge base. The process involves:
    1) Embedding documents into vectors, 2) Storing them in a vector database,
    3) Retrieving relevant documents for a query, 4) Providing them as context to the LLM.
    This allows LLMs to answer questions about specific documents or private data.
    """,
]


def example_1_basic_rag():
    """
    Example 1: Basic RAG Pipeline
    Complete RAG from scratch
    """
    print_section(
        "Example 1: Basic RAG Pipeline",
        "Build a complete RAG system step by step"
    )
    
    print_step(1, "Prepare documents")
    
    # Create Document objects
    documents = [Document(page_content=doc.strip()) for doc in SAMPLE_DOCS]
    print(f"   Loaded {len(documents)} documents")
    
    print_step(2, "Split documents into chunks")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"   Created {len(splits)} chunks")
    print(f"   Sample chunk: {splits[0].page_content[:100]}...")
    
    print_step(3, "Create embeddings and vector store")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("   ✅ Vector store created (using free HuggingFace embeddings)")
    
    print_step(4, "Create retriever")
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Return top 3 most relevant chunks
    )
    print("   Retriever configured (k=3)")
    
    print_step(5, "Build QA chain")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    print_step(6, "Ask questions")
    
    questions = [
        "What is LangChain?",
        "How do agents work in LangChain?",
        "What is RAG and how does it work?",
    ]
    
    for question in questions:
        print(f"\n   ❓ Question: {question}")
        result = qa_chain.invoke({"query": question})
        
        print(f"   💡 Answer: {result['result']}")
        print(f"\n   📚 Sources Used ({len(result['source_documents'])} chunks):")
        for i, doc in enumerate(result['source_documents'], 1):
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"      {i}. {preview}...")
        input("\n      Press Enter for next question...")


def example_2_custom_rag_chain():
    """
    Example 2: Custom RAG Chain with LCEL
    Build a custom RAG chain using LangChain Expression Language
    """
    print_section(
        "Example 2: Custom RAG Chain",
        "Build a customizable RAG chain with LCEL"
    )
    
    print_step(1, "Setup vector store")
    
    documents = [Document(page_content=doc.strip()) for doc in SAMPLE_DOCS]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    print_step(2, "Create custom prompt")
    
    template = """Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based solely on the context above.
If the context doesn't contain the answer, say "I don't have enough information to answer that."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    print_step(3, "Build custom chain with LCEL")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    # Format documents helper
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("""
    Chain structure:
    question → retriever → format_docs → prompt → llm → parser → answer
    """)
    
    print_step(4, "Test the custom chain")
    
    questions = [
        "What is LangGraph?",
        "Explain memory in LangChain",
    ]
    
    for question in questions:
        print(f"\n   ❓ Question: {question}")
        
        # Get the answer
        answer = rag_chain.invoke(question)
        print_response(answer, "Answer")
        
        # Show the sources used
        docs = retriever.get_relevant_documents(question)
        print(f"\n   📚 Sources Used ({len(docs)} chunks):")
        for i, doc in enumerate(docs, 1):
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"      {i}. {preview}...")
        
        input("\n      Press Enter for next question...")


def example_3_rag_with_sources():
    """
    Example 3: RAG with Source Citations
    Track and display which documents were used
    """
    print_section(
        "Example 3: RAG with Source Citations",
        "Answer questions and cite sources"
    )
    
    print_step(1, "Create documents with metadata")
    
    docs_with_metadata = [
        Document(
            page_content=SAMPLE_DOCS[0].strip(),
            metadata={"source": "LangChain Intro", "page": 1}
        ),
        Document(
            page_content=SAMPLE_DOCS[1].strip(),
            metadata={"source": "LangGraph Guide", "page": 1}
        ),
        Document(
            page_content=SAMPLE_DOCS[2].strip(),
            metadata={"source": "Chains Tutorial", "page": 2}
        ),
        Document(
            page_content=SAMPLE_DOCS[3].strip(),
            metadata={"source": "Agents Guide", "page": 3}
        ),
        Document(
            page_content=SAMPLE_DOCS[4].strip(),
            metadata={"source": "Memory Documentation", "page": 4}
        ),
        Document(
            page_content=SAMPLE_DOCS[5].strip(),
            metadata={"source": "RAG Tutorial", "page": 5}
        ),
    ]
    
    print(f"   Created {len(docs_with_metadata)} documents with metadata")
    
    print_step(2, "Build vector store")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs_with_metadata, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    print_step(3, "Create QA chain with sources")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    def format_docs_with_sources(docs):
        """Format documents with source information."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    template = """Based on the following sources, answer the question.
Cite which source number(s) you used.

{context}

Question: {question}

Answer (include source citations):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print_step(4, "Ask questions with source tracking")
    
    questions = [
        "What are the different types of memory?",
        "How does RAG work?",
    ]
    
    for question in questions:
        print(f"\n   ❓ Question: {question}")
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(question)
        
        # Get answer
        answer = rag_chain.invoke(question)
        
        print_response(answer, "Answer with Citations")
        
        print("\n   📚 Retrieved Sources:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:100] + "..."
            print(f"      {i}. {source} (Page {page})")
            print(f"         {preview}\n")
        
        input("      Press Enter for next question...")


def example_4_semantic_search():
    """
    Example 4: Semantic Search
    Explore similarity search capabilities
    """
    print_section(
        "Example 4: Semantic Search",
        "Understanding vector similarity search"
    )
    
    print_step(1, "Create vector store")
    
    documents = [Document(page_content=doc.strip()) for doc in SAMPLE_DOCS]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print_step(2, "Perform similarity search")
    
    query = "Tell me about AI agents"
    print(f"\n   🔍 Query: {query}\n")
    
    # Search for similar documents
    similar_docs = vectorstore.similarity_search(query, k=3)
    
    print("   📊 Most Similar Documents:\n")
    for i, doc in enumerate(similar_docs, 1):
        print(f"   {i}. {doc.page_content[:150]}...")
        print()
    
    print_step(3, "Similarity search with scores")
    
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    
    print("\n   📊 Documents with Similarity Scores:\n")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"   {i}. Score: {score:.4f}")
        print(f"      {doc.page_content[:100]}...\n")
    
    print("   💡 Lower scores = more similar")


def example_5_advanced_rag():
    """
    Example 5: Advanced RAG Techniques
    Multiple retrievers and reranking
    """
    print_section(
        "Example 5: Advanced RAG",
        "Advanced techniques: MMR and filtering"
    )
    
    print_step(1, "Setup vector store")
    
    documents = [Document(page_content=doc.strip()) for doc in SAMPLE_DOCS]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print_step(2, "Use MMR (Maximum Marginal Relevance)")
    
    print("\n   MMR balances relevance with diversity\n")
    
    query = "LangChain features"
    
    # Regular similarity search
    print("   📊 Regular similarity search:")
    regular_docs = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(regular_docs, 1):
        print(f"   {i}. {doc.page_content[:80]}...")
    
    # MMR search
    print("\n   📊 MMR search (more diverse):")
    mmr_docs = vectorstore.max_marginal_relevance_search(query, k=3)
    for i, doc in enumerate(mmr_docs, 1):
        print(f"   {i}. {doc.page_content[:80]}...")
    
    print_step(3, "Create retriever with custom search")
    
    # Create retriever with MMR
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 6,  # Fetch more candidates before MMR
            "lambda_mult": 0.5  # Balance between relevance and diversity
        }
    )
    
    print("""
    MMR Parameters:
    - k: Number of final results
    - fetch_k: Number of candidates to consider
    - lambda_mult: 0 = max diversity, 1 = max relevance
    """)
    
    print_step(4, "Build RAG chain with MMR")
    
    llm = ChatOpenAI(model=settings.default_model, temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    template = """Answer based on context:

{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": mmr_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    question = "What are the main concepts in LangChain?"
    print(f"\n   ❓ Question: {question}")
    
    answer = rag_chain.invoke(question)
    print_response(answer, "Answer (using MMR)")
    
    # Show the sources used
    docs = mmr_retriever.get_relevant_documents(question)
    print(f"\n   📚 Sources Used with MMR ({len(docs)} chunks):")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"      {i}. {preview}...")


def main():
    """Run all examples."""
    if not validate_api_keys():
        return
    
    print("\n" + "="*70)
    print(" RAG (RETRIEVAL AUGMENTED GENERATION) - DEMO".center(70))
    print("="*70 + "\n")
    
    try:
        example_1_basic_rag()
        input("\nPress Enter to continue to next example...")
        
        example_2_custom_rag_chain()
        input("\nPress Enter to continue to next example...")
        
        example_3_rag_with_sources()
        input("\nPress Enter to continue to next example...")
        
        example_4_semantic_search()
        input("\nPress Enter to continue to next example...")
        
        example_5_advanced_rag()
        
        print("\n" + "="*70)
        print(" ✅ All RAG examples completed!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

