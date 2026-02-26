"""
Module de gestion des embeddings pour la vectorisation des événements.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EventEmbeddingManager:
    """Gestionnaire pour créer des embeddings à partir des descriptions d'événements."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialiser le gestionnaire d'embeddings.
        
        Args:
            model_name: Nom du modèle HuggingFace à utiliser
        """
        logger.info(f"Loading embeddings model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.model_dimension = 384  # Dimension de ce modèle
        logger.info(f"Model loaded. Dimension: {self.model_dimension}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Créer un embedding pour un texte.
        
        Args:
            text: Texte à vectoriser
            
        Returns:
            Vecteur numpy
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text for embedding: {text}")
            return None
        
        try:
            embedding = self.embeddings.embed_query(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None

    def embed_events(self, events: List[Dict]) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Créer des embeddings pour une liste d'événements.
        
        Args:
            events: Liste des événements avec 'title' et 'description'
            
        Returns:
            Tuple (embeddings, events_with_valid_embeddings)
        """
        embeddings = []
        valid_events = []
        
        for i, event in enumerate(events):
            # Créer un texte combiné titre + description
            text = f"{event.get('title', '')} {event.get('description', '')}"
            text = text.strip()
            
            if not text:
                logger.warning(f"Event {i} has no text content, skipping")
                continue
            
            embedding = self.embed_text(text)
            if embedding is not None:
                embeddings.append(embedding)
                valid_events.append(event)
        
        logger.info(f"Created {len(embeddings)} embeddings out of {len(events)} events")
        return embeddings, valid_events

    def get_dimension(self) -> int:
        """Retourner la dimension des embeddings."""
        return self.model_dimension
