"""
Core Legal-AI implementation
"""

import os
import yaml
from typing import List, Dict, Optional
from dotenv import load_dotenv
from loguru import logger

from .parsers import GovInfoParser
from .rag import LegalRAG


class LegalAI:
    """Main Legal-AI class for document analysis and RAG operations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Legal-AI system"""
        load_dotenv()
        self.config = self._load_config(config_path)
        self.parser = GovInfoParser(self.config)
        self.rag = LegalRAG(self.config)
        self.documents = []
        self.index = None
        
        # Setup logging
        self._setup_logging()
        logger.info("Legal-AI system initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "llama_index": {
                "chunk_size": 1024,
                "chunk_overlap": 200,
                "similarity_top_k": 5
            },
            "rag": {
                "temperature": 0.1,
                "max_tokens": 512
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_file = os.getenv("LOG_FILE", "legal_ai.log")
        
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            rotation="10 MB"
        )
    
    def load_document(self, url: str) -> None:
        """Load and parse a single government document"""
        logger.info(f"Loading document from: {url}")
        try:
            documents = self.parser.parse_govinfo_document(url)
            self.documents.extend(documents)
            logger.info(f"Successfully loaded {len(documents)} document chunks")
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
            raise
    
    def load_documents(self, urls: List[str]) -> None:
        """Load and parse multiple government documents"""
        logger.info(f"Loading {len(urls)} documents")
        for url in urls:
            self.load_document(url)
    
    def build_index(self) -> None:
        """Build the RAG index from loaded documents"""
        if not self.documents:
            raise ValueError("No documents loaded. Load documents first.")
        
        logger.info("Building RAG index...")
        self.index = self.rag.build_index(self.documents)
        logger.info("RAG index built successfully")
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        if self.index is None:
            self.build_index()
        
        logger.info(f"Processing query: {question}")
        try:
            response = self.rag.query(question, self.index)
            logger.info("Query processed successfully")
            return response
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def compare_documents(self, query: str) -> str:
        """Compare loaded documents based on a query"""
        if len(self.documents) < 2:
            raise ValueError("Need at least 2 documents for comparison")
        
        comparison_prompt = f"""
        Compare the following documents based on this question: {query}
        
        Analyze the similarities, differences, and key insights across the documents.
        """
        
        return self.query(comparison_prompt)
    
    def get_document_summary(self) -> Dict:
        """Get summary of loaded documents"""
        return {
            "total_documents": len(self.documents),
            "has_index": self.index is not None,
            "config": self.config
        }
