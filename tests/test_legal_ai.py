"""
Tests for Legal-AI functionality
"""

import pytest
import os
from unittest.mock import Mock, patch
from legal_ai.parsers import GovInfoParser
from legal_ai.rag import RAGSystem, LegalRAG
from legal_ai.core import LegalAI
from llama_index import Document


class TestGovInfoParser:
    """Test cases for GovInfoParser"""
    
    def test_parser_initialization(self):
        """Test parser initialization"""
        parser = GovInfoParser()
        assert parser.base_url == "https://www.govinfo.gov"
        assert parser.rate_limit == 10
    
    def test_document_type_extraction(self):
        """Test document type extraction from URL"""
        parser = GovInfoParser()
        
        test_cases = [
            ("https://www.govinfo.gov/content/pkg/CDOC-119hdoc6/html/CDOC-119hdoc6.htm", "congressional_document"),
            ("https://www.govinfo.gov/content/pkg/BILLS-118hr1/html/BILLS-118hr1ih.htm", "bill"),
            ("https://www.govinfo.gov/content/pkg/CRPT-118hrpt1/html/CRPT-118hrpt1.htm", "congressional_report"),
        ]
        
        for url, expected_type in test_cases:
            result = parser._extract_document_type(url)
            assert result == expected_type
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        parser = GovInfoParser()
        
        dirty_text = """
        
        
        This is a test.
        
        
        Another line.
        
        
        """
        
        cleaned = parser._clean_text(dirty_text)
        expected = "This is a test.\nAnother line."
        assert cleaned == expected
    
    @patch('requests.get')
    def test_fetch_document_success(self, mock_get):
        """Test successful document fetching"""
        mock_response = Mock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        parser = GovInfoParser()
        content = parser.fetch_document("https://example.com/test")
        
        assert content == "<html><body>Test content</body></html>"
        mock_get.assert_called_once()
    
    def test_chunk_document(self):
        """Test document chunking"""
        parser = GovInfoParser({"llama_index": {"chunk_size": 50, "chunk_overlap": 10}})
        
        text = "This is a test document. " * 10  # ~250 characters
        chunks = parser.chunk_document(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)  # Including overlap


class TestRAGSystem:
    """Test cases for RAG system"""
    
    def test_rag_initialization(self):
        """Test RAG system initialization"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            rag = RAGSystem()
            assert rag.config == {}
            assert rag.service_context is not None
    
    @patch('legal_ai.rag.VectorStoreIndex.from_documents')
    def test_build_index(self, mock_from_documents):
        """Test index building"""
        mock_index = Mock()
        mock_from_documents.return_value = mock_index
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            rag = RAGSystem()
            documents = [Document(text="Test document")]
            
            result = rag.build_index(documents)
            
            assert result == mock_index
            mock_from_documents.assert_called_once()


class TestLegalRAG:
    """Test cases for LegalRAG"""
    
    def test_legal_content_classification(self):
        """Test legal content classification"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            legal_rag = LegalRAG()
            
            test_cases = [
                ("Section 1. This is a legal section.", "structured_legal_text"),
                ("The budget appropriation for this year is $1M.", "budget_document"),
                ("Whereas the congress finds...", "legislative_text"),
                ("This is general content.", "general_legal"),
            ]
            
            for text, expected in test_cases:
                result = legal_rag._classify_legal_content(text)
                assert result == expected
    
    def test_legal_entity_extraction(self):
        """Test legal entity extraction"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            legal_rag = LegalRAG()
            
            text = "Section 1 provides for $1,000,000 in funding. Section 2 allocates $500.00 more."
            entities = legal_rag._extract_legal_entities(text)
            
            assert "Section 1" in entities
            assert "Section 2" in entities
            assert "$1,000,000" in entities
            assert "$500.00" in entities


class TestLegalAI:
    """Test cases for main LegalAI class"""
    
    def test_legal_ai_initialization(self):
        """Test LegalAI initialization"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            legal_ai = LegalAI()
            assert legal_ai.documents == []
            assert legal_ai.index is None
    
    def test_default_config_loading(self):
        """Test default config when file doesn't exist"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            legal_ai = LegalAI(config_path="nonexistent.yaml")
            
            assert legal_ai.config["llama_index"]["chunk_size"] == 1024
            assert legal_ai.config["rag"]["temperature"] == 0.1
    
    def test_document_summary(self):
        """Test document summary generation"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            legal_ai = LegalAI()
            
            summary = legal_ai.get_document_summary()
            
            assert "total_documents" in summary
            assert "has_index" in summary
            assert "config" in summary
            assert summary["total_documents"] == 0
            assert summary["has_index"] is False


# Integration tests
class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.integration
    @patch('requests.get')
    def test_full_pipeline_mock(self, mock_get):
        """Test full pipeline with mocked HTTP requests"""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.text = """
        <html>
        <head><title>Test Congressional Document</title></head>
        <body>
        <div class="document-content">
        <h1>Congressional Document Test</h1>
        <p>This is a test congressional document about budget allocations.</p>
        <p>Section 1 provides for $1,000,000 in defense spending.</p>
        <p>Section 2 allocates $500,000 for education programs.</p>
        </div>
        </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            # Test the full pipeline
            legal_ai = LegalAI()
            
            # Load document
            url = "https://www.govinfo.gov/content/pkg/CDOC-119hdoc6/html/CDOC-119hdoc6.htm"
            legal_ai.load_document(url)
            
            # Verify documents were loaded
            assert len(legal_ai.documents) > 0
            
            # Check document content
            first_doc = legal_ai.documents[0]
            assert "congressional document" in first_doc.text.lower()
            assert first_doc.metadata["document_type"] == "congressional_document"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
