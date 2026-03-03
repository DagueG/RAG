"""
Module pour construire et gérer l'index Faiss.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.clean_data import EventDataCleaner
from src.vectorization.embeddings import EventEmbeddingManager

logger = logging.getLogger(__name__)


class FaissIndexBuilder:
    """Constructeur et gestionnaire de l'index Faiss."""

    def __init__(self):
        """Initialiser le builder."""
        self.index = None
        self.embeddings_manager = None
        self.events = None
        self.embedding_dimension = None

    def build_index(self, events: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
        """
        Construire un index Faiss à partir d'événements.
        
        Args:
            events: Liste des événements nettoyés
            
        Returns:
            Tuple (index, events_with_embeddings)
        """
        logger.info(f"Building Faiss index for {len(events)} events")
        
        # Initialiser le gestionnaire d'embeddings
        self.embeddings_manager = EventEmbeddingManager()
        self.embedding_dimension = self.embeddings_manager.get_dimension()
        
        # Créer les embeddings
        embeddings, valid_events = self.embeddings_manager.embed_events(events)
        
        if not embeddings:
            logger.error("No valid embeddings created")
            return None, []
        
        # Convertir en numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        logger.info(f"Embeddings shape: {embeddings_array.shape}")
        
        # Créer l'index Faiss
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.index.add(embeddings_array)
        self.events = valid_events
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
        return self.index, valid_events

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Chercher les k événements les plus proches d'une requête.
        
        Args:
            query_embedding: Vecteur de la requête
            k: Nombre de résultats à retourner
            
        Returns:
            Liste des événements les plus proches
        """
        if self.index is None or self.events is None:
            logger.error("Index not built yet")
            return []
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        distances, indices = self.index.search(query_embedding, k=min(k, self.index.ntotal))
        
        results = []
        for idx in indices[0]:
            if idx >= 0:  # -1 means not found
                results.append(self.events[idx])
        
        return results

    def save_index(self, output_dir: Path) -> None:
        """
        Sauvegarder l'index et les métadonnées.
        
        Args:
            output_dir: Répertoire de sortie
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.index is None:
            logger.error("Index not built yet")
            return
        
        # Sauvegarder l'index
        index_path = output_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Index saved to {index_path}")
        
        # Sauvegarder les événements
        events_path = output_dir / "events_metadata.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2)
        logger.info(f"Events metadata saved to {events_path}")
        
        # Sauvegarder les informations du modèle
        metadata_path = output_dir / "index_metadata.pkl"
        metadata = {
            "embedding_dimension": self.embedding_dimension,
            "num_events": len(self.events),
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved to {metadata_path}")

    @staticmethod
    def load_index(index_dir: Path) -> Tuple[faiss.Index, List[Dict], Dict]:
        """
        Charger un index sauvegardé.
        
        Args:
            index_dir: Répertoire contenant l'index
            
        Returns:
            Tuple (index, events, metadata)
        """
        index_dir = Path(index_dir)
        
        # Charger l'index
        index_path = index_dir / "faiss_index.bin"
        index = faiss.read_index(str(index_path))
        logger.info(f"Index loaded from {index_path}")
        
        # Charger les événements
        events_path = index_dir / "events_metadata.json"
        with open(events_path, "r", encoding="utf-8") as f:
            events = json.load(f)
        logger.info(f"Loaded {len(events)} events")
        
        # Charger les métadonnées
        metadata_path = index_dir / "index_metadata.pkl"
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        logger.info(f"Metadata loaded: {metadata}")
        
        return index, events, metadata


def build_full_index(processed_data_path: Path, output_dir: Path) -> None:
    """
    Fonction utilitaire pour construire un index complet.
    
    Args:
        processed_data_path: Chemin du fichier CSV/JSON des événements nettoyés
        output_dir: Répertoire de sortie pour l'index
    """
    # Charger les données
    cleaner = EventDataCleaner()
    if processed_data_path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(processed_data_path)
        events = df.to_dict("records")
    else:
        events = cleaner.load_raw_events(processed_data_path)
    
    logger.info(f"Loaded {len(events)} events from {processed_data_path}")
    
    # Construire l'index
    builder = FaissIndexBuilder()
    index, valid_events = builder.build_index(events)
    
    if index is None:
        logger.error("Failed to build index")
        return
    
    # Sauvegarder
    builder.save_index(output_dir)
    logger.info(f"Index successfully created with {index.ntotal} vectors")

if __name__ == "__main__":
    """Script principal pour construire l'index Faiss."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Déterminer le chemin des données nettoyées
    base_dir = Path(__file__).parent.parent.parent
    processed_data = base_dir / "data" / "processed" / "toulouse_events.json"
    output_dir = base_dir / "data" / "faiss_index"
    
    logger.info(f"Building index from {processed_data}")
    logger.info(f"Output directory: {output_dir}")
    
    if not processed_data.exists():
        logger.error(f"File not found: {processed_data}")
        logger.info("Please run: python src/data_processing/clean_data.py")
        sys.exit(1)
    
    build_full_index(processed_data, output_dir)
    logger.info("✅ Index build complete!")