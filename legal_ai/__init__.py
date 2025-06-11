"""
Legal-AI: Government Document Analysis with RAG
"""

__version__ = "1.0.0"
__author__ = "Legal-AI Team"

from .parsers import GovInfoParser, DocumentParser
from .rag import RAGSystem, LegalRAG
from .core import LegalAI

__all__ = [
    "GovInfoParser",
    "DocumentParser", 
    "RAGSystem",
    "LegalRAG",
    "LegalAI"
]
