"""
RAG Chain: Orchestrates vector search + LLM generation for event recommendations.
Combines Faiss index with Mistral LLM to provide intelligent responses.
"""

import logging
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

from mistralai.client import MistralClient
from langchain_core.prompts import PromptTemplate

from src.vectorization.embeddings import EventEmbeddingManager
from src.vectorization.build_index import FaissIndexBuilder

logger = logging.getLogger(__name__)


class RAGChain:
    """
    RAG system combining Faiss vector search with Mistral LLM.
    
    Workflow:
    1. User asks question in French
    2. Question is vectorized using EventEmbeddingManager
    3. Faiss searches for top-k similar events
    4. Retrieved events + question sent to Mistral
    5. LLM generates conversational French response
    """
    
    def __init__(
        self,
        index_dir: str = "data/faiss_index",
        model_name: str = "mistral-small",
        top_k: int = 5,
        api_key: str = None
    ):
        """
        Initialize RAG chain.
        
        Args:
            index_dir: Path to Faiss index directory
            model_name: Mistral model to use
            top_k: Number of events to retrieve by default
            api_key: Mistral API key (from env if None)
        """
        logger.info("Initializing RAG Chain...")
        
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.top_k = top_k
        
        # Get API key from argument or environment
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY")
            if api_key:
                logger.info("Loaded MISTRAL_API_KEY from environment")
        
        logger.info(f"API key provided: {bool(api_key)}")
        if api_key:
            logger.info(f"API key (first 10 chars): {api_key[:10]}")
        
        # Initialize Mistral client (store reference, may be None in tests)
        self.mistral_api_key = api_key
        self.mistral_client = None
        try:
            if api_key:  # Only initialize if we have a key
                logger.info("Attempting to initialize MistralClient...")
                self.mistral_client = MistralClient(api_key=api_key)
                logger.info("Mistral client initialized with API key")
            else:
                logger.warning("No API key provided - RAG will not generate responses")
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            logger.info("Continuing without Mistral client - will be mocked in tests")
        
        # Initialize embedding manager
        self.embedding_manager = EventEmbeddingManager()
        logger.info("Embedding manager initialized")
        
        # Initialize Faiss index builder and load index
        self.index_builder = FaissIndexBuilder()
        try:
            # Load index from disk - load_index is a static method that returns (index, events, metadata)
            index, events, metadata = FaissIndexBuilder.load_index(str(self.index_dir))
            # Store in instance for access via search()
            self.index_builder.index = index
            self.index_builder.events = events
            logger.info(f"Faiss index loaded from {self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to load Faiss index: {e}")
            raise
        
        # Load events metadata from the loaded events
        self.events_metadata = events  # Use the events loaded from Faiss index
        logger.info(f"Loaded {len(self.events_metadata)} events metadata")
        
        # Define prompt template for LLM
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Vous êtes un assistant expert en événements culturels à Toulouse.

Voici une liste d'événements culturels pertinents:
{context}

Basé sur ces informations, répondez à la question suivante en français de manière concise et conversationnelle:

Question: {question}

Réponse:"""
        )
        
        logger.info("RAG Chain initialized successfully")
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load events metadata from saved JSON file."""
        metadata_path = self.index_dir / "events_metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found at {metadata_path}")
            return []
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return []
    
    def search_events(self, query: str, k: int = None) -> Tuple[List[Dict], List[float]]:
        """
        Search for relevant events using vector similarity.
        
        Args:
            query: User question in French
            k: Number of results (uses self.top_k if None)
        
        Returns:
            Tuple of (events_list, distances) - distances are empty list since FaissIndexBuilder doesn't return them
        """
        if k is None:
            k = self.top_k
        
        logger.info(f"Searching for {k} events matching: {query[:50]}...")
        
        try:
            # Vectorize query
            query_embedding = self.embedding_manager.embed_text(query)
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            
            # Search in Faiss index - returns list of event dictionaries
            events = self.index_builder.search(query_embedding, k)
            
            logger.info(f"Found {len(events)} relevant events")
            # Return events and empty distances list (distances not provided by search)
            return events, [0.0] * len(events)
        
        except Exception as e:
            logger.error(f"Error searching events: {e}")
            return [], []
    
    def _format_context(self, events: List[Dict]) -> str:
        """Format retrieved events into readable context."""
        if not events:
            return "Aucun événement trouvé correspondant à votre recherche."
        
        context_lines = []
        for i, event in enumerate(events, 1):
            context = f"\n{i}. **{event.get('title', 'Sans titre')}**"
            context += f"\n   Date: {event.get('date_start', 'Date inconnue')}"
            context += f"\n   Lieu: {event.get('location', 'Lieu inconnu')}"
            
            # Add description if available
            description = event.get('description', '')
            if description:
                desc_short = description[:200] + '...' if len(description) > 200 else description
                context += f"\n   Description: {desc_short}"
            
            context += f"\n   URL: {event.get('url', 'Non disponible')}"
            context_lines.append(context)
        
        return "\n".join(context_lines)
    
    def generate_response(
        self,
        query: str,
        k: int = None,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Generate intelligent response to user query using RAG.
        
        Args:
            query: User question in French
            k: Number of events to retrieve (uses self.top_k if None)
            include_context: Include retrieved events in response metadata
        
        Returns:
            Dictionary with 'response', 'events', 'distances', 'query'
        """
        logger.info(f"Generating response for: {query}")
        
        if k is None:
            k = self.top_k
        
        # Search for relevant events
        events, distances = self.search_events(query, k)
        
        # Format context from retrieved events
        context = self._format_context(events)
        
        # Create prompt
        prompt_text = self.prompt_template.format(context=context, question=query)
        
        try:
            # Call Mistral to generate response
            logger.debug("Calling Mistral LLM...")
            response = self.mistral_client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                max_tokens=500
            )
            
            generated_text = response.choices[0].message.content
            logger.info("Response generated successfully")
            
            return {
                "response": generated_text,
                "query": query,
                "events": events if include_context else None,
                "distances": distances if include_context else None,
                "model": self.model_name,
                "num_events_retrieved": len(events)
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def reload_index(self) -> bool:
        """Reload the Faiss index (useful after index rebuild)."""
        try:
            logger.info(f"Reloading index from {self.index_dir}...")
            # Load index from disk
            index, events, metadata = FaissIndexBuilder.load_index(str(self.index_dir))
            # Store in instance
            self.index_builder.index = index
            self.index_builder.events = events
            self.events_metadata = events
            logger.info("Index reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload index: {e}")
            return False
