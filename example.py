#!/usr/bin/env python3
"""
Example script demonstrating Legal-AI functionality
"""

import os
import sys
from pathlib import Path

# Add the legal_ai package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from legal_ai import LegalAI
from loguru import logger


def main():
    """Main example function"""
    print("🏛️  Legal-AI: Government Document Analysis Demo")
    print("=" * 50)
    
    try:
        # Initialize Legal-AI
        print("1. Initializing Legal-AI system...")
        legal_ai = LegalAI()
        
        # Example government document URL
        doc_url = "https://www.govinfo.gov/content/pkg/CDOC-119hdoc6/html/CDOC-119hdoc6.htm"
        
        print(f"2. Loading document from govinfo.gov...")
        print(f"   URL: {doc_url}")
        legal_ai.load_document(doc_url)
        
        print("3. Building RAG index...")
        legal_ai.build_index()
        
        print("4. Running example queries...")
        print("-" * 30)
        
        # Example queries
        queries = [
            "What is the main purpose of this congressional document?",
            "What are the key budget allocations mentioned?",
            "Who are the main stakeholders or agencies involved?",
            "What are the timeline and implementation details?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print("Answer:", end=" ")
            
            try:
                response = legal_ai.query(query)
                print(response[:200] + "..." if len(response) > 200 else response)
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        
        # Display system summary
        summary = legal_ai.get_document_summary()
        print(f"\nSystem Summary:")
        print(f"- Documents loaded: {summary['total_documents']}")
        print(f"- Index built: {summary['has_index']}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have set up your .env file with API keys")
        print("2. Check your internet connection")
        print("3. Verify that all dependencies are installed")
        return 1
    
    return 0


def batch_processing_example():
    """Example of batch processing multiple documents"""
    print("\n🔄 Batch Processing Example")
    print("-" * 30)
    
    legal_ai = LegalAI()
    
    # Multiple document URLs
    urls = [
        "https://www.govinfo.gov/content/pkg/CDOC-119hdoc6/html/CDOC-119hdoc6.htm",
        # Add more URLs as needed
    ]
    
    print(f"Loading {len(urls)} documents...")
    legal_ai.load_documents(urls)
    
    print("Performing comparative analysis...")
    comparison = legal_ai.compare_documents(
        "What are the common themes across these documents?"
    )
    
    print("Comparison Result:")
    print(comparison[:300] + "..." if len(comparison) > 300 else comparison)


def interactive_mode():
    """Interactive query mode"""
    print("\n💬 Interactive Mode")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    legal_ai = LegalAI()
    
    # Load a sample document
    doc_url = "https://www.govinfo.gov/content/pkg/CDOC-119hdoc6/html/CDOC-119hdoc6.htm"
    print(f"Loading document: {doc_url}")
    legal_ai.load_document(doc_url)
    legal_ai.build_index()
    
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("Processing...")
            response = legal_ai.query(query)
            print(f"\nAnswer: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found")
        print("Please copy .env.example to .env and add your API keys")
        print("")
    
    # Run main demo
    exit_code = main()
    
    # Optional: Run additional examples
    if exit_code == 0:
        choice = input("\nWould you like to try interactive mode? (y/n): ").lower()
        if choice == 'y':
            interactive_mode()
    
    sys.exit(exit_code)
