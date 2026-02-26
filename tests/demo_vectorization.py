"""
Script de démonstration de la vectorisation et indexation Faiss.
"""

import json
import logging
from pathlib import Path

from src.vectorization.embeddings import EventEmbeddingManager
from src.vectorization.build_index import FaissIndexBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_vectorization():
    """Démonstration du pipeline de vectorisation."""
    
    print("=" * 60)
    print("DEMO: Vectorization and Faiss Indexing")
    print("=" * 60)
    
    # Créer des événements d'exemple
    print("\n[1] Loading events...")
    base_dir = Path(__file__).parent.parent
    processed_data = base_dir / "data" / "processed" / "toulouse_events_demo.json"
    
    if not processed_data.exists():
        print(f"   ERROR: {processed_data} not found!")
        print("   Please run: python -m tests.demo_data_processing")
        return
    
    with open(processed_data, "r", encoding="utf-8") as f:
        events = json.load(f)
    
    print(f"   OK Loaded {len(events)} events")
    
    # Créer les embeddings
    print("\n[2] Creating embeddings...")
    manager = EventEmbeddingManager()
    embeddings, valid_events = manager.embed_events(events)
    print(f"   OK Created {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {manager.get_dimension()}")
    
    # Construire l'index
    print("\n[3] Building Faiss index...")
    builder = FaissIndexBuilder()
    index, index_events = builder.build_index(events)
    print(f"   OK Index created with {index.ntotal} vectors")
    
    # Sauvegarder l'index
    print("\n[4] Saving index...")
    index_dir = base_dir / "data" / "faiss_index"
    builder.save_index(index_dir)
    print(f"   OK Index saved to {index_dir}")
    
    # Test des recherches
    print("\n[5] Testing similarity search...")
    
    # Requête 1: Jazz
    query1 = manager.embed_text("Je cherche un concert de jazz")
    results1 = builder.search(query1, k=2)
    
    print(f"\n   Query: 'Je cherche un concert de jazz'")
    for i, event in enumerate(results1, 1):
        print(f"   Result {i}: {event['title']}")
    
    # Requête 2: Art
    query2 = manager.embed_text("Exposition d'art moderne")
    results2 = builder.search(query2, k=2)
    
    print(f"\n   Query: 'Exposition d'art moderne'")
    for i, event in enumerate(results2, 1):
        print(f"   Result {i}: {event['title']}")
    
    # Requête 3: Theatre
    query3 = manager.embed_text("Festival de theatre")
    results3 = builder.search(query3, k=2)
    
    print(f"\n   Query: 'Festival de theatre'")
    for i, event in enumerate(results3, 1):
        print(f"   Result {i}: {event['title']}")
    
    print("\n" + "=" * 60)
    print("OK Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo_vectorization()
