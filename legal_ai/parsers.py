"""
Document parsers for government documents from govinfo.gov
"""

import requests
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from loguru import logger
from llama_index import Document
import html2text


class GovInfoParser:
    """Parser for government documents from govinfo.gov"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.base_url = "https://www.govinfo.gov"
        self.rate_limit = self.config.get("govinfo", {}).get("rate_limit", 10)
        self.retry_attempts = self.config.get("govinfo", {}).get("retry_attempts", 3)
        self.retry_delay = self.config.get("govinfo", {}).get("retry_delay", 1)
        
        # HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        
    def parse_govinfo_document(self, url: str) -> List[Document]:
        """Parse a government document from govinfo.gov URL"""
        logger.info(f"Parsing document from: {url}")
        
        # Fetch the document content
        content = self.fetch_document(url)
        
        # Extract metadata
        metadata = self.extract_metadata(content, url)
        
        # Parse HTML content into text
        text_content = self.parse_html_content(content)
        
        # Chunk the document
        chunks = self.chunk_document(text_content)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "source_url": url
            })
            
            documents.append(Document(
                text=chunk,
                metadata=doc_metadata
            ))
        
        logger.info(f"Created {len(documents)} document chunks")
        return documents
    
    def fetch_document(self, url: str) -> str:
        """Fetch document content from URL with retries"""
        for attempt in range(self.retry_attempts):
            try:
                # Rate limiting
                time.sleep(1 / self.rate_limit)
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                logger.info(f"Successfully fetched document from {url}")
                return response.text
                
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch document after {self.retry_attempts} attempts")
                    raise
    
    def extract_metadata(self, html_content: str, url: str) -> Dict:
        """Extract metadata from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        metadata = {
            "source_url": url,
            "document_type": self._extract_document_type(url),
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # Extract document-specific information
        self._extract_document_info(soup, metadata)
        
        return metadata
    
    def _extract_document_type(self, url: str) -> str:
        """Extract document type from URL"""
        path = urlparse(url).path
        if '/CDOC-' in path:
            return 'congressional_document'
        elif '/BILLS-' in path:
            return 'bill'
        elif '/CRPT-' in path:
            return 'congressional_report'
        elif '/CREC-' in path:
            return 'congressional_record'
        elif '/FR-' in path:
            return 'federal_register'
        else:
            return 'unknown'
    
    def _extract_document_info(self, soup: BeautifulSoup, metadata: Dict):
        """Extract additional document information"""
        # Look for document header information
        header = soup.find('div', class_='document-header')
        if header:
            metadata["header_info"] = header.get_text().strip()
        
        # Extract date information
        date_elements = soup.find_all(['span', 'div'], class_=lambda x: x and 'date' in x.lower())
        for elem in date_elements:
            metadata.setdefault("dates", []).append(elem.get_text().strip())
    
    def parse_html_content(self, html_content: str) -> str:
        """Convert HTML content to plain text"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract main content
        main_content = soup.find('div', class_='document-content')
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Convert to text using html2text for better formatting
            text = self.html_converter.handle(str(main_content))
        else:
            text = self.html_converter.handle(html_content)
        
        # Clean up the text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        cleaned_lines = []
        
        for line in lines:
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def chunk_document(self, text: str) -> List[str]:
        """Split document into chunks for processing"""
        chunk_size = self.config.get("llama_index", {}).get("chunk_size", 1024)
        chunk_overlap = self.config.get("llama_index", {}).get("chunk_overlap", 200)
        
        # Simple chunking by character count
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks


class DocumentParser(GovInfoParser):
    """Alias for GovInfoParser for backward compatibility"""
    pass


class BatchProcessor:
    """Process multiple documents in batch"""
    
    def __init__(self, config: Dict = None):
        self.parser = GovInfoParser(config)
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """Process multiple URLs and return all documents"""
        all_documents = []
        
        for i, url in enumerate(urls):
            logger.info(f"Processing document {i+1}/{len(urls)}: {url}")
            try:
                documents = self.parser.parse_govinfo_document(url)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                continue
        
        logger.info(f"Processed {len(urls)} URLs, created {len(all_documents)} document chunks")
        return all_documents


class EnhancedGovInfoParser(GovInfoParser):
    """Enhanced parser with additional features"""
    
    def parse_with_metadata(self, url: str) -> Dict:
        """Parse document and return both content and detailed metadata"""
        content = self.fetch_document(url)
        metadata = self.extract_metadata(content, url)
        documents = self.parse_govinfo_document(url)
        
        return {
            "documents": documents,
            "metadata": metadata,
            "raw_content": content,
            "url": url
        }
