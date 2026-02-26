"""
Demo script for RAG Chain: Demonstrates the complete RAG pipeline.
Shows how to ask questions about cultural events and get AI-generated responses.
"""
# -*- coding: utf-8 -*-

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 encoding for proper French output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.rag_chain import RAGChain


def print_separator(title: str = ""):
    """Print formatted separator."""
    if title:
        print(f"\n{'=' * 60}")
        print(title)
        print('=' * 60)
    else:
        print(f"\n{'-' * 60}\n")


def print_result(result: dict) -> None:
    """Format and print RAG result."""
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nEvents retrieved: {result['num_events_retrieved']}")
    
    if result['events'] and len(result['events']) > 0:
        print("\nContext (Events used):")
        for i, event in enumerate(result['events'], 1):
            print(f"\n  [{i}] {event.get('title', 'Sans titre')}")
            print(f"      Date: {event.get('date_start', 'Unknown')}")
            print(f"      Location: {event.get('location', 'Unknown')}")


def main():
    """Run RAG Chain demo with sample questions."""
    
    print_separator("DEMO: RAG Chain - Cultural Event Chatbot")
    
    try:
        # Initialize RAG Chain
        print("\n[1] Initializing RAG Chain...")
        rag_chain = RAGChain(
            index_dir="data/faiss_index",
            top_k=3,
            model_name="mistral-small"
        )
        print("    OK RAG Chain initialized")
        
    except Exception as e:
        print(f"    ERROR Failed to initialize RAG Chain: {e}")
        print("    Make sure to run Étape 3 tests first to generate the Faiss index")
        return
    
    # Sample questions for demo
    sample_questions = [
        "Je cherche un concert de jazz ou de musique classique",
        "Quels événements y a-t-il au musée d'art?",
        "Y a-t-il un festival de théâtre de prévu?"
    ]
    
    print(f"\n[2] Testing with {len(sample_questions)} sample queries...")
    print("    Queries will be sent to Mistral LLM for response generation")
    print("    (This may take a moment on first run as the model loads...)")
    
    for i, question in enumerate(sample_questions, 1):
        print_separator(f"Query {i}/{len(sample_questions)}")
        
        try:
            # Generate response using RAG
            if rag_chain.mistral_client is None:
                print("\n    [!] Mistral client not available (no API key)")
                print("    [1] Performing search anyway...")
                events, distances = rag_chain.search_events(question, k=3)
                print(f"    [2] Search found {len(events)} events")
                if events:
                    print("\n    Events retrieved:")
                    for j, event in enumerate(events, 1):
                        print(f"        {j}. {event.get('title', 'Untitled')}")
                print("\n    [!] In production, Mistral would generate a response here")
                print("    For full demo, set MISTRAL_API_KEY environment variable")
            else:
                result = rag_chain.generate_response(
                    query=question,
                    k=3,
                    include_context=True
                )
                
                # Display result
                print_result(result)
            
        except Exception as e:
            print(f"\nERROR Processing query: {e}")
            if "NoneType" in str(e):
                print("Make sure Mistral API is accessible and your environment is properly configured")
                print("You may need to set MISTRAL_API_KEY environment variable")
    
    print_separator("Demo Summary")
    print("\nRAG Chain demonstration completed!")
    print("\nKey components tested:")
    print("  [1] Query vectorization (EventEmbeddingManager)")
    print("  [2] Semantic search (Faiss Index)")
    print("  [3] Context formatting")
    print("  [4] LLM response generation (Mistral)")
    print("\nNext steps:")
    print("  - Run: pytest tests/test_rag_chain.py -v")
    print("  - Then proceed to Étape 5 (FastAPI REST API)")
    print_separator()


if __name__ == "__main__":
    main()
