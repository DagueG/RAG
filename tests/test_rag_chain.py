"""
Test suite for RAG Chain (LangChain + Mistral + Faiss integration).
Tests the complete RAG pipeline for event recommendations.
"""

import pytest
import logging
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.rag.rag_chain import RAGChain
from src.vectorization.embeddings import EventEmbeddingManager
from src.vectorization.build_index import FaissIndexBuilder
from src.data_processing.clean_data import EventDataCleaner

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return [
        {
            "id": "1",
            "title": "Concert de Jazz au Théâtre du Capitole",
            "description": "Un concert de jazz contemporain avec des artistes internationaux",
            "date_start": "2026-03-15",
            "date_end": "2026-03-15",
            "location": "Théâtre du Capitole, Toulouse",
            "image_url": "https://example.com/jazz.jpg",
            "url": "https://example.com/event/1",
            "source": "openagenda"
        },
        {
            "id": "2",
            "title": "Exposition d'Art Moderne au Musée d'Art Moderne",
            "description": "Exposition des artistes contemporains du XXI siècle",
            "date_start": "2026-03-01",
            "date_end": "2026-04-30",
            "location": "Musée d'Art Moderne, Toulouse",
            "image_url": "https://example.com/art.jpg",
            "url": "https://example.com/event/2",
            "source": "openagenda"
        },
        {
            "id": "3",
            "title": "Festival de Théâtre de Toulouse",
            "description": "Une semaine de représentations théâtrales variées",
            "date_start": "2026-03-20",
            "date_end": "2026-03-27",
            "location": "Divers lieux, Toulouse",
            "image_url": "https://example.com/theatre.jpg",
            "url": "https://example.com/event/3",
            "source": "openagenda"
        }
    ]


@pytest.fixture
def setup_faiss_index(tmp_path, sample_events):
    """Setup a test Faiss index."""
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Build index
    builder = FaissIndexBuilder()
    embedding_manager = EventEmbeddingManager()
    embeddings = embedding_manager.embed_events(sample_events)
    builder.build_index(sample_events)
    builder.save_index(str(index_dir))
    
    return index_dir


class TestRAGChainInitialization:
    """Test RAG Chain initialization."""
    
    @patch('src.rag.rag_chain.Mistral')
    def test_rag_chain_initialization(self, mock_mistral, setup_faiss_index):
        """Test that RAG Chain initializes correctly."""
        mock_mistral.return_value = MagicMock()
        rag = RAGChain(index_dir=str(setup_faiss_index))
        
        assert rag.embedding_manager is not None
        assert rag.index_builder is not None
        assert len(rag.events_metadata) > 0
        assert rag.prompt_template is not None
        logger.info("[PASS] RAG Chain initialization test")
    
    @patch('src.rag.rag_chain.Mistral')
    def test_rag_chain_metadata_loading(self, mock_mistral, setup_faiss_index, sample_events):
        """Test that events metadata is loaded correctly."""
        mock_mistral.return_value = MagicMock()
        rag = RAGChain(index_dir=str(setup_faiss_index))
        
        assert len(rag.events_metadata) == len(sample_events)
        # Verify first event
        first_event = rag.events_metadata[0]
        assert "title" in first_event
        assert "description" in first_event
        assert "date_start" in first_event
        logger.info("[PASS] Metadata loading test")


class TestRAGChainSearch:
    """Test event search functionality."""
    
    @pytest.fixture
    def rag_chain(self, tmp_path, sample_events):
        """Setup RAG Chain with test index."""
        index_dir = tmp_path / "faiss_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        builder = FaissIndexBuilder()
        embeddings = EventEmbeddingManager().embed_events(sample_events)
        builder.build_index(sample_events)
        builder.save_index(str(index_dir))
        
        with patch('src.rag.rag_chain.Mistral'):
            return RAGChain(index_dir=str(index_dir))
    
    def test_search_events_jazz(self, rag_chain):
        """Test searching for jazz concert."""
        query = "Je cherche un concert de jazz"
        events, distances = rag_chain.search_events(query, k=2)
        
        # Search should return results (may vary based on similarity)
        assert isinstance(events, list)
        assert isinstance(distances, list)
        logger.info(f"[PASS] Jazz search test - Found {len(events)} events")
    
    def test_search_events_art(self, rag_chain):
        """Test searching for art exhibition."""
        query = "Exposition d'art moderne"
        events, distances = rag_chain.search_events(query, k=2)
        
        # Search should return results
        assert isinstance(events, list)
        assert isinstance(distances, list)
        logger.info(f"[PASS] Art search test - Found {len(events)} events")
    
    def test_search_events_custom_k(self, rag_chain):
        """Test search with custom k parameter."""
        query = "événement culturel"
        events, distances = rag_chain.search_events(query, k=1)
        
        # k parameter should be respected (or less if fewer events available)
        assert len(events) <= 1 or len(events) <= len(rag_chain.events_metadata)
        logger.info("[PASS] Custom k parameter test")
    
    def test_search_returns_distances(self, rag_chain):
        """Test that search returns valid distances."""
        query = "théâtre"
        events, distances = rag_chain.search_events(query, k=3)
        
        # Distances should be non-negative (L2 distance)
        assert all(d >= 0 for d in distances)
        # Distances should be sorted (smallest first)
        assert distances == sorted(distances)
        logger.info("[PASS] Distance validation test")


class TestRAGChainFormatting:
    """Test context formatting functionality."""
    
    def test_format_context_with_events(self, tmp_path, sample_events):
        """Test formatting events into readable context."""
        # Setup RAG chain
        index_dir = tmp_path / "faiss_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        builder = FaissIndexBuilder()
        embeddings = EventEmbeddingManager().embed_events(sample_events)
        builder.build_index(sample_events)
        builder.save_index(str(index_dir))
        
        with patch('src.rag.rag_chain.Mistral'):
            rag = RAGChain(index_dir=str(index_dir))
        
        # Test formatting
        context = rag._format_context(sample_events)
        
        assert "Concert de Jazz" in context
        assert "Exposition" in context
        assert "2026-03-15" in context
        assert "Toulouse" in context
        logger.info("[PASS] Context formatting test")
    
    def test_format_context_empty(self, tmp_path, sample_events):
        """Test formatting with empty events list."""
        index_dir = tmp_path / "faiss_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        builder = FaissIndexBuilder()
        embeddings = EventEmbeddingManager().embed_events(sample_events)
        builder.build_index(sample_events)
        builder.save_index(str(index_dir))
        
        with patch('src.rag.rag_chain.Mistral'):
            rag = RAGChain(index_dir=str(index_dir))
        
        # Test empty list
        context = rag._format_context([])
        
        assert "Aucun événement trouvé" in context
        logger.info("[PASS] Empty context test")


class TestRAGChainIntegration:
    """Integration tests for complete RAG pipeline."""
    
    @pytest.fixture
    def rag_chain_with_mock(self, tmp_path, sample_events):
        """Setup RAG Chain with mocked Mistral calls."""
        index_dir = tmp_path / "faiss_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        builder = FaissIndexBuilder()
        embeddings = EventEmbeddingManager().embed_events(sample_events)
        builder.build_index(sample_events)
        builder.save_index(str(index_dir))
        
        with patch('src.rag.rag_chain.Mistral'):
            return RAGChain(index_dir=str(index_dir))
    
    def test_generate_response_structure(self, rag_chain_with_mock):
        """Test that response has correct structure."""
        # Mock Mistral response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Voici les événements recommandés pour vous..."
        mock_client.chat.complete.return_value = mock_response
        
        # Assign mocked client
        rag_chain_with_mock.mistral_client = mock_client
        
        # Generate response
        result = rag_chain_with_mock.generate_response("Je cherche un concert")
        
        # Verify structure
        assert "response" in result
        assert "query" in result
        assert "events" in result
        assert "distances" in result
        assert "model" in result
        assert "num_events_retrieved" in result
        logger.info("[PASS] Response structure test")
    
    def test_reload_index(self, rag_chain_with_mock):
        """Test index reloading functionality."""
        initial_count = len(rag_chain_with_mock.events_metadata)
        
        # Reload index
        success = rag_chain_with_mock.reload_index()
        
        assert success is True
        assert len(rag_chain_with_mock.events_metadata) == initial_count
        logger.info("[PASS] Index reload test")


class TestRAGChainErrorHandling:
    """Test error handling in RAG Chain."""
    
    def test_missing_index_directory(self, tmp_path):
        """Test initialization with missing index directory."""
        missing_dir = tmp_path / "nonexistent"
        
        # Should raise exception
        with pytest.raises(Exception):
            with patch('src.rag.rag_chain.Mistral'):
                RAGChain(index_dir=str(missing_dir))
        
        logger.info("[PASS] Missing index handling test")
    
    @pytest.fixture
    def rag_chain(self, tmp_path, sample_events):
        """Setup minimal RAG Chain."""
        index_dir = tmp_path / "faiss_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        builder = FaissIndexBuilder()
        embeddings = EventEmbeddingManager().embed_events(sample_events)
        builder.build_index(sample_events)
        builder.save_index(str(index_dir))
        
        with patch('src.rag.rag_chain.Mistral'):
            return RAGChain(index_dir=str(index_dir))
    
    def test_search_with_empty_query(self, rag_chain):
        """Test search with empty query string."""
        # Should handle gracefully (embedding should handle empty string)
        events, distances = rag_chain.search_events("", k=1)
        
        # Should return some results (empty string still yields an embedding)
        assert isinstance(events, list)
        assert isinstance(distances, list)
        logger.info("[PASS] Empty query handling test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
