"""
Test suite for FastAPI RAG endpoints.
Tests all API routes with mocked RAG responses.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
import sys
from io import StringIO

from src.api.main import app, QuestionRequest, RebuildRequest

logger = logging.getLogger(__name__)

# Create client inside test scope to avoid initialization issues
@pytest.fixture
def client():
    """Create a TestClient for testing."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test /health endpoint."""
        with patch('src.api.main.rag_chain', MagicMock()):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "rag_initialized" in data
            logger.info("[PASS] Health check test")


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self, client):
        """Test / root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        logger.info("[PASS] Root endpoint test")


class TestInfoEndpoint:
    """Test info endpoint."""
    
    def test_info_with_rag(self, client):
        """Test /info endpoint with initialized RAG."""
        mock_rag = MagicMock()
        mock_rag.events_metadata = [{"id": "1"}, {"id": "2"}]
        mock_rag.model_name = "mistral-small"
        
        with patch('src.api.main.rag_chain', mock_rag):
            response = client.get("/info")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["events_indexed"] == 2
            assert data["model"] == "mistral-small"
            logger.info("[PASS] Info endpoint test")


class TestAskEndpoint:
    """Test /ask endpoint for question answering."""
    
    def test_ask_valid_question(self, client):
        """Test /ask with valid question."""
        mock_rag = MagicMock()
        mock_rag.generate_response.return_value = {
            "response": "Voici les événements...",
            "num_events_retrieved": 3,
            "events": [
                {
                    "id": "1",
                    "title": "Concert",
                    "date_start": "2026-03-15",
                    "location": "Toulouse"
                }
            ],
            "model": "mistral-small"
        }
        
        with patch('src.api.main.rag_chain', mock_rag):
            response = client.post(
                "/ask",
                json={
                    "question": "Je cherche un concert",
                    "k": 5
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Voici les événements..."
            assert data["query"] == "Je cherche un concert"
            assert data["events_retrieved"] == 3
            logger.info("[PASS] Ask valid question test")
    
    def test_ask_empty_question(self, client):
        """Test /ask with empty question."""
        mock_rag = MagicMock()
        
        with patch('src.api.main.rag_chain', mock_rag):
            response = client.post(
                "/ask",
                json={"question": "", "k": 5}
            )
            
            assert response.status_code == 400
            logger.info("[PASS] Empty question rejection test")
    
    def test_ask_no_rag(self, client):
        """Test /ask when RAG is not initialized."""
        with patch('src.api.main.rag_chain', None):
            response = client.post(
                "/ask",
                json={"question": "Je cherche un événement"}
            )
            
            assert response.status_code == 503
            logger.info("[PASS] No RAG initialization test")
    
    def test_ask_with_custom_k(self, client):
        """Test /ask with custom k parameter."""
        mock_rag = MagicMock()
        mock_rag.generate_response.return_value = {
            "response": "Response...",
            "num_events_retrieved": 2,
            "events": [],
            "model": "mistral-small"
        }
        
        with patch('src.api.main.rag_chain', mock_rag):
            response = client.post(
                "/ask",
                json={
                    "question": "Quelle question?",
                    "k": 2
                }
            )
            
            assert response.status_code == 200
            # Verify k was passed to generate_response
            mock_rag.generate_response.assert_called_once()
            call_args = mock_rag.generate_response.call_args
            assert call_args[1]["k"] == 2
            logger.info("[PASS] Custom k parameter test")


class TestRebuildEndpoint:
    """Test /rebuild endpoint for index management."""
    
    def test_rebuild_success(self, client):
        """Test /rebuild with successful index reload."""
        mock_rag = MagicMock()
        mock_rag.reload_index.return_value = True
        mock_rag.events_metadata = [
            {"id": "1"},
            {"id": "2"},
            {"id": "3"}
        ]
        
        with patch('src.api.main.rag_chain', mock_rag):
            response = client.post(
                "/rebuild",
                json={"index_dir": "data/faiss_index"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["events_indexed"] == 3
            logger.info("[PASS] Rebuild success test")
    
    def test_rebuild_no_rag(self, client):
        """Test /rebuild when RAG is not initialized."""
        with patch('src.api.main.rag_chain', None):
            response = client.post(
                "/rebuild",
                json={"index_dir": "data/faiss_index"}
            )
            
            assert response.status_code == 503
            logger.info("[PASS] Rebuild no RAG test")
    
    def test_rebuild_failure(self, client):
        """Test /rebuild when reload fails."""
        mock_rag = MagicMock()
        mock_rag.reload_index.return_value = False
        
        with patch('src.api.main.rag_chain', mock_rag):
            response = client.post(
                "/rebuild",
                json={"index_dir": "data/faiss_index"}
            )
            
            assert response.status_code == 500
            logger.info("[PASS] Rebuild failure test")


class TestEndpointIntegration:
    """Integration tests for multiple endpoints."""
    
    def test_full_workflow(self, client):
        """Test a full workflow: health -> info -> ask."""
        mock_rag = MagicMock()
        mock_rag.events_metadata = [{"id": "1"}, {"id": "2"}]
        mock_rag.model_name = "mistral-small"
        mock_rag.generate_response.return_value = {
            "response": "Voici les événements...",
            "num_events_retrieved": 2,
            "events": [{"id": "1", "title": "Event"}],
            "model": "mistral-small"
        }
        
        with patch('src.api.main.rag_chain', mock_rag):
            # Check health
            health = client.get("/health")
            assert health.status_code == 200
            
            # Get info
            info = client.get("/info")
            assert info.status_code == 200
            assert info.json()["events_indexed"] == 2
            
            # Ask question
            ask = client.post(
                "/ask",
                json={"question": "Quel événement?"}
            )
            assert ask.status_code == 200
            
            logger.info("[PASS] Full workflow test")


class TestResponseModels:
    """Test Pydantic response models."""
    
    def test_question_request_model(self):
        """Test QuestionRequest model validation."""
        # Valid request
        req = QuestionRequest(question="Test?", k=5)
        assert req.question == "Test?"
        assert req.k == 5
        
        # Default k
        req2 = QuestionRequest(question="Test?")
        assert req2.k == 5
        logger.info("[PASS] QuestionRequest model test")
    
    def test_rebuild_request_model(self):
        """Test RebuildRequest model."""
        req = RebuildRequest(index_dir="data/faiss_index")
        assert req.index_dir == "data/faiss_index"
        
        # Default index_dir
        req2 = RebuildRequest()
        assert req2.index_dir == "data/faiss_index"
        logger.info("[PASS] RebuildRequest model test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
