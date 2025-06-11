"""
RAG (Retrieval-Augmented Generation) system for legal documents
"""

import os
from typing import List, Dict, Optional
from loguru import logger

from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import Document
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode


class RAGSystem:
    """Basic RAG system implementation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.service_context = self._create_service_context()
        self.index = None
        self.query_engine = None
    
    def _create_service_context(self) -> ServiceContext:
        """Create LlamaIndex service context"""
        # Configure LLM
        llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=self.config.get("rag", {}).get("temperature", 0.1),
            max_tokens=self.config.get("rag", {}).get("max_tokens", 512)
        )
        
        # Configure embedding model
        embed_model = OpenAIEmbedding()
        
        return ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            chunk_size=self.config.get("llama_index", {}).get("chunk_size", 1024),
            chunk_overlap=self.config.get("llama_index", {}).get("chunk_overlap", 200)
        )
    
    def build_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build vector index from documents"""
        logger.info(f"Building index from {len(documents)} documents")
        
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.get("llama_index", {}).get("similarity_top_k", 5),
            response_mode=ResponseMode.COMPACT
        )
        
        logger.info("Index built successfully")
        return self.index
    
    def query(self, question: str, index: VectorStoreIndex = None) -> str:
        """Query the RAG system"""
        if index:
            query_engine = index.as_query_engine(
                similarity_top_k=self.config.get("llama_index", {}).get("similarity_top_k", 5)
            )
        elif self.query_engine:
            query_engine = self.query_engine
        else:
            raise ValueError("No index available. Build index first.")
        
        logger.info(f"Processing query: {question}")
        response = query_engine.query(question)
        
        return str(response)
    
    def get_sources(self, response_text: str) -> List[str]:
        """Extract source information from response"""
        # This is a simplified implementation
        # In practice, you'd extract this from the response metadata
        return ["Sources available in response metadata"]


class LegalRAG(RAGSystem):
    """Specialized RAG system for legal documents"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.legal_prompt_template = self._create_legal_prompt_template()
    
    def _create_legal_prompt_template(self) -> str:
        """Create specialized prompt template for legal documents"""
        return """
        You are a legal AI assistant analyzing government and legal documents.
        
        Context: {context_str}
        
        Question: {query_str}
        
        Instructions:
        1. Provide accurate information based only on the provided context
        2. Include relevant legal citations and references when available
        3. If the information is not in the context, clearly state this
        4. Structure your response in a clear, professional manner
        5. Highlight key legal concepts and terminology
        
        Response:
        """
    
    def build_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build index with legal document enhancements"""
        # Add legal-specific metadata processing
        enhanced_documents = self._enhance_legal_documents(documents)
        
        return super().build_index(enhanced_documents)
    
    def _enhance_legal_documents(self, documents: List[Document]) -> List[Document]:
        """Add legal-specific enhancements to documents"""
        enhanced = []
        
        for doc in documents:
            # Add legal document type classification
            doc.metadata["legal_category"] = self._classify_legal_content(doc.text)
            
            # Extract legal entities (simplified)
            doc.metadata["legal_entities"] = self._extract_legal_entities(doc.text)
            
            enhanced.append(doc)
        
        return enhanced
    
    def _classify_legal_content(self, text: str) -> str:
        """Classify the type of legal content"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ["section", "subsection", "paragraph"]):
            return "structured_legal_text"
        elif any(term in text_lower for term in ["budget", "appropriation", "funding"]):
            return "budget_document"
        elif any(term in text_lower for term in ["whereas", "resolved", "enacted"]):
            return "legislative_text"
        else:
            return "general_legal"
    
    def _extract_legal_entities(self, text: str) -> List[str]:
        """Extract legal entities from text (simplified implementation)"""
        # This is a basic implementation - in practice you'd use NER
        entities = []
        
        # Look for common legal patterns
        import re
        
        # Find section references
        sections = re.findall(r'Section \d+', text, re.IGNORECASE)
        entities.extend(sections)
        
        # Find dollar amounts
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        entities.extend(amounts)
        
        return list(set(entities))
    
    def create_query_engine(self, index: VectorStoreIndex):
        """Create specialized query engine for legal documents"""
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.config.get("llama_index", {}).get("similarity_top_k", 5)
        )
        
        return RetrieverQueryEngine(retriever=retriever)


class MultiModalLegalRAG(LegalRAG):
    """Multi-modal RAG for documents with tables, images, etc."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
    
    def build_multimodal_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build index supporting multi-modal content"""
        # Process documents for tables and structured content
        processed_documents = self._process_multimodal_content(documents)
        
        return self.build_index(processed_documents)
    
    def _process_multimodal_content(self, documents: List[Document]) -> List[Document]:
        """Process documents to extract tables and structured content"""
        processed = []
        
        for doc in documents:
            # Extract table information (simplified)
            if "table" in doc.text.lower():
                doc.metadata["contains_table"] = True
                doc.metadata["table_info"] = self._extract_table_info(doc.text)
            
            processed.append(doc)
        
        return processed
    
    def _extract_table_info(self, text: str) -> Dict:
        """Extract table information from text"""
        return {
            "has_numeric_data": any(char.isdigit() for char in text),
            "estimated_rows": text.count('\n'),
            "content_type": "tabular_data"
        }
