# Legal-AI: Government Document Analysis with RAG

Legal-AI is an AI-powered platform that streamlines legal research, automates contract review, and analyzes legal documents. This project demonstrates how to parse government documents from govinfo.gov using LlamaIndex and build a Retrieval-Augmented Generation (RAG) system for intelligent legal document analysis.

## Features

- 🔍 **Government Document Parsing**: Extract and process legal documents from govinfo.gov
- 🤖 **RAG Implementation**: Retrieval-Augmented Generation for intelligent document querying
- 📚 **LlamaIndex Integration**: Advanced indexing and retrieval capabilities
- 🏛️ **Legal Document Analysis**: Specialized processing for congressional documents, bills, and legal texts
- 💡 **Semantic Search**: Find relevant information across large legal document collections

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Government Document Parsing](#government-document-parsing)
- [Building the RAG System](#building-the-rag-system)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key (or other LLM provider)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/loponly/legal-ai.git
   cd legal-ai
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Quick Start

Here's a simple example to get you started with parsing a government document and building a RAG system:

```python
from legal_ai import DocumentParser, RAGSystem
import os

# Initialize the document parser
parser = DocumentParser()

# Parse a government document from govinfo.gov
doc_url = "https://www.govinfo.gov/content/pkg/CDOC-119hdoc6/html/CDOC-119hdoc6.htm"
documents = parser.parse_govinfo_document(doc_url)

# Build the RAG system
rag = RAGSystem()
index = rag.build_index(documents)

# Query the system
response = rag.query("What are the main findings of this congressional document?")
print(response)
```

## Government Document Parsing

### Supported Document Types

- Congressional Documents (CDOC)
- Bills (BILLS)
- Congressional Reports (CRPT)
- Congressional Records (CREC)
- Federal Register documents (FR)

### Parsing Examples

#### 1. Parse Congressional Document

```python
from legal_ai.parsers import GovInfoParser
from llama_index import Document

# Initialize parser
parser = GovInfoParser()

# Parse specific congressional document
url = "https://www.govinfo.gov/content/pkg/CDOC-119hdoc6/html/CDOC-119hdoc6.htm"
content = parser.fetch_document(url)
documents = parser.parse_html_content(content)

print(f"Extracted {len(documents)} document chunks")
```

#### 2. Batch Processing Multiple Documents

```python
from legal_ai.parsers import BatchProcessor

# Define document URLs
urls = [
    "https://www.govinfo.gov/content/pkg/CDOC-119hdoc6/html/CDOC-119hdoc6.htm",
    "https://www.govinfo.gov/content/pkg/BILLS-118hr1/html/BILLS-118hr1ih.htm",
    # Add more URLs as needed
]

# Process all documents
processor = BatchProcessor()
all_documents = processor.process_urls(urls)
```

#### 3. Advanced Parsing with Metadata

```python
from legal_ai.parsers import EnhancedGovInfoParser

parser = EnhancedGovInfoParser()
result = parser.parse_with_metadata(url)

# Access parsed content and metadata
documents = result['documents']
metadata = result['metadata']

print(f"Document Title: {metadata['title']}")
print(f"Publication Date: {metadata['date']}")
print(f"Document Type: {metadata['type']}")
```

## Building the RAG System

### 1. Basic RAG Implementation

```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from legal_ai.rag import LegalRAG

# Configure service context
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Initialize Legal RAG system
legal_rag = LegalRAG(service_context=service_context)

# Build index from parsed documents
index = legal_rag.build_index(documents)

# Create query engine
query_engine = legal_rag.create_query_engine(index)
```

### 2. Advanced RAG with Custom Retrievers

```python
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from legal_ai.retrievers import LegalDocumentRetriever

# Custom retriever for legal documents
retriever = LegalDocumentRetriever(
    index=index,
    similarity_top_k=5,
    filters={"document_type": "congressional"}
)

# Create query engine with custom retriever
query_engine = RetrieverQueryEngine(retriever=retriever)
```

### 3. Multi-Modal RAG for Complex Documents

```python
from legal_ai.rag import MultiModalLegalRAG

# Handle documents with tables, images, and complex structures
mm_rag = MultiModalLegalRAG()
mm_index = mm_rag.build_multimodal_index(documents)

# Query with context awareness
response = mm_rag.query(
    "Analyze the budget allocations in Table 3 of the congressional document",
    index=mm_index
)
```

## Usage Examples

### Example 1: Analyzing Congressional Budget Documents

```python
from legal_ai import LegalAI

# Initialize the system
legal_ai = LegalAI()

# Load and analyze a budget document
budget_doc = "https://www.govinfo.gov/content/pkg/CDOC-119hdoc6/html/CDOC-119hdoc6.htm"
legal_ai.load_document(budget_doc)

# Ask specific questions
questions = [
    "What is the total budget allocation for defense spending?",
    "Which programs received the largest funding increases?",
    "What are the projected costs for infrastructure projects?"
]

for question in questions:
    answer = legal_ai.query(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Example 2: Comparative Analysis

```python
# Compare multiple versions of legislation
legal_ai.load_documents([
    "https://www.govinfo.gov/content/pkg/BILLS-118hr1/html/BILLS-118hr1ih.htm",
    "https://www.govinfo.gov/content/pkg/BILLS-118hr1/html/BILLS-118hr1rh.htm"
])

# Perform comparative analysis
comparison = legal_ai.compare_documents(
    query="What are the key differences between the introduced and reported versions?"
)
print(comparison)
```

### Example 3: Legal Research Assistant

```python
from legal_ai.assistants import LegalResearchAssistant

# Create a specialized assistant
assistant = LegalResearchAssistant()

# Research specific legal topics
research_query = "Find all references to environmental protection regulations"
results = assistant.research(research_query)

# Generate summary report
report = assistant.generate_report(results)
print(report)
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Alternative LLM providers
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key

# Vector Store Configuration
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env

# Logging
LOG_LEVEL=INFO
LOG_FILE=legal_ai.log
```

### Configuration File

Create `config.yaml`:

```yaml
# LlamaIndex Configuration
llama_index:
  chunk_size: 1024
  chunk_overlap: 200
  similarity_top_k: 5

# Document Processing
document_processing:
  max_document_size: 10MB
  supported_formats: ["html", "pdf", "txt"]
  
# RAG Configuration
rag:
  temperature: 0.1
  max_tokens: 512
  response_mode: "compact"

# Parsing Settings
parsing:
  extract_tables: true
  extract_images: false
  preserve_formatting: true
```

## API Reference

### DocumentParser Class

```python
class DocumentParser:
    def parse_govinfo_document(self, url: str) -> List[Document]:
        """Parse a government document from govinfo.gov"""
        
    def extract_metadata(self, content: str) -> Dict:
        """Extract metadata from document content"""
        
    def chunk_document(self, content: str) -> List[str]:
        """Split document into manageable chunks"""
```

### RAGSystem Class

```python
class RAGSystem:
    def build_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build vector index from documents"""
        
    def query(self, question: str) -> str:
        """Query the RAG system"""
        
    def get_sources(self, response: str) -> List[str]:
        """Get source documents for a response"""
```

## Advanced Features

### 1. Custom Document Processors

```python
from legal_ai.processors import CustomDocumentProcessor

class CongressionalDocumentProcessor(CustomDocumentProcessor):
    def process(self, content: str) -> List[Document]:
        # Custom logic for congressional documents
        sections = self.extract_sections(content)
        return [Document(text=section) for section in sections]

# Register custom processor
legal_ai.register_processor("congressional", CongressionalDocumentProcessor())
```

### 2. Intelligent Caching

```python
from legal_ai.cache import DocumentCache

# Enable caching for faster repeated queries
cache = DocumentCache(cache_dir="./cache")
legal_ai.set_cache(cache)

# Documents are automatically cached after first retrieval
```

### 3. Real-time Document Monitoring

```python
from legal_ai.monitoring import GovInfoMonitor

# Monitor for new documents
monitor = GovInfoMonitor()
monitor.watch_collection("CDOC")  # Congressional Documents
monitor.on_new_document(callback=legal_ai.auto_index)
```

## Performance Optimization

### 1. Parallel Processing

```python
from legal_ai.parallel import ParallelProcessor

# Process multiple documents in parallel
processor = ParallelProcessor(max_workers=4)
results = processor.process_batch(document_urls)
```

### 2. Memory Management

```python
# Configure for large document collections
legal_ai.configure_memory(
    max_memory_gb=8,
    enable_swapping=True,
    chunk_processing=True
)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_parsing.py
pytest tests/test_rag.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=legal_ai tests/
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "legal_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

```bash
# Deploy to AWS Lambda
serverless deploy

# Deploy to Google Cloud Run
gcloud run deploy legal-ai --source .
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -m 'Add some feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: support@legal-ai.com
- 💬 Discord: [Join our community](https://discord.gg/legal-ai)
- 📖 Documentation: [docs.legal-ai.com](https://docs.legal-ai.com)
- 🐛 Issues: [GitHub Issues](https://github.com/loponly/legal-ai/issues)

## Author

**Enkhbat.E**  
Email: enkhbat@em4it.com

## Acknowledgments

- [LlamaIndex](https://docs.llamaindex.ai/) for the powerful indexing and retrieval framework
- [govinfo.gov](https://www.govinfo.gov/) for providing access to government documents
- The open-source community for their valuable contributions

---

**Note**: This project is for educational and research purposes. Ensure compliance with govinfo.gov terms of service and applicable laws when using this software.

