"""
Demo script for RAG API.
Shows how to interact with the FastAPI endpoints.
"""

# -*- coding: utf-8 -*-

import sys
import io
from pathlib import Path

# Force UTF-8 encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

# Sample queries to test
SAMPLE_QUERIES = [
    "Je cherche un concert de jazz ou de musique classique",
    "Quels événements y a-t-il au musée d'art?",
    "Y a-t-il un festival de théâtre de prévu?"
]


def print_separator(title: str = ""):
    """Print formatted separator."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)
    else:
        print('=' * 60)


def check_health():
    """Check API health status."""
    print_separator("1. HEALTH CHECK")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: {data['status']}")
            print(f"✓ RAG Initialized: {data['rag_initialized']}")
            return True
        else:
            print(f"✗ Health check failed (HTTP {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API")
        print(f"  Make sure the API is running on {BASE_URL}")
        print("  Run in another terminal: python -m uvicorn src.api.main:app --reload")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def get_info():
    """Get system information."""
    print_separator("2. SYSTEM INFO")
    try:
        response = requests.get(f"{BASE_URL}/info")
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status')}")
            print(f"Events indexed: {data.get('events_indexed', 0)}")
            print(f"Model: {data.get('model')}")
            print(f"Embedding dimension: {data.get('vector_dimension')}")
            return True
        else:
            print(f"✗ Info request failed (HTTP {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def ask_question(question: str, k: int = 5):
    """Ask a question to the RAG system."""
    print(f"\nQuery: {question}")
    print('-' * 60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={
                "question": question,
                "k": k
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Print response
            print(f"\nResponse:")
            print(data['response'])
            
            # Print retrieved events
            print(f"\nEvents retrieved: {data['events_retrieved']}")
            print(f"Model used: {data['model_used']}")
            
            if data['events']:
                print("\nContext (Events used):")
                for i, event in enumerate(data['events'], 1):
                    print(f"\n  [{i}] {event.get('title', 'N/A')}")
                    print(f"      Date: {event.get('date_start', 'N/A')}")
                    print(f"      Location: {event.get('location', 'N/A')}")
            
            return True
        
        elif response.status_code == 400:
            print(f"✗ Bad request: {response.json().get('detail')}")
            return False
        
        elif response.status_code == 503:
            print(f"✗ Service unavailable: {response.json().get('detail')}")
            return False
        
        else:
            print(f"✗ Error (HTTP {response.status_code}): {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print("✗ Request timeout (API took too long)")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def rebuild_index():
    """Rebuild the Faiss index."""
    print_separator("REBUILD INDEX")
    print("Rebuilding index... (this may take a moment)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/rebuild",
            json={"index_dir": "data/faiss_index"},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Rebuild successful")
            print(f"  Events indexed: {data['events_indexed']}")
            return True
        else:
            print(f"✗ Rebuild failed (HTTP {response.status_code})")
            print(f"  {response.json().get('detail')}")
            return False
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Main demo flow."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " RAG System API Demo".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Check health
    if not check_health():
        print("\n✗ API is not accessible. Exiting.")
        return
    
    # Get info
    if not get_info():
        print("\n✗ Could not get system info.")
        return
    
    # Test with sample queries
    print_separator("3. TESTING WITH SAMPLE QUERIES")
    print(f"Testing with {len(SAMPLE_QUERIES)} sample queries...\n")
    
    successful = 0
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n[{i}/{len(SAMPLE_QUERIES)}]")
        print("=" * 60)
        if ask_question(query):
            successful += 1
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print_separator("DEMO SUMMARY")
    print(f"\n✓ API is operational and responding")
    print(f"✓ Successfully tested {successful}/{len(SAMPLE_QUERIES)} queries")
    print(f"\nAPI Documentation available at: {BASE_URL}/docs")
    print(f"OpenAPI spec at: {BASE_URL}/openapi.json")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
