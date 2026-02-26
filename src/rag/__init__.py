"""
RAG (Retrieval-Augmented Generation) module for cultural event recommendations.
Combines Faiss vector search with Mistral LLM for intelligent responses.
"""

from src.rag.rag_chain import RAGChain

__all__ = ["RAGChain"]
