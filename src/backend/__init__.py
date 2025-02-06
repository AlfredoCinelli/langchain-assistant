"""Package containing the main modules to perform RAG.

It contains two main modules:

* core: with the main RAG building blocks:
    - VectorStorePinecone: to manage the Pinecone vector store and retrieve documents
    - LLMModel: to manage the LLM model and get an instace of it
    - RAGChain: to manage the RAG chain and strem a full chain.
    
* insgestion: to ingest the scraped LangChain docs into the Pinecone vector store.
"""

from .core import LLMModel, RAGChain, VectorStorePinecone, RAGChain2

__all__ = [
    "LLMModel",
    "RAGChain",
    "VectorStorePinecone",
    "RAGChain2",
]