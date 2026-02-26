"""
Quick integration test for RAG Chain with actual Mistral API.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.rag.rag_chain import RAGChain

# Test
api_key = os.getenv('MISTRAL_API_KEY')
print(f"API Key loaded: {bool(api_key)}")
print(f"API Key value (first 10 chars): {api_key[:10] if api_key else 'None'}")

try:
    rag = RAGChain(
        index_dir="data/faiss_index",
        api_key=api_key
    )
    
    print(f"RAG Chain initialized: {rag.mistral_client is not None}")
    
    if rag.mistral_client:
        # Try a simple search
        query = "Je cherche un concert de jazz"
        result = rag.generate_response(query)
        print(f"\nQuery: {query}")
        print(f"Response: {result['response'][:200]}...")
    else:
        print("Mistral client is None - no API key")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
