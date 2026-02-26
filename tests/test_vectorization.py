"""
Tests unitaires pour la vectorisation et l'indexation Faiss.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.vectorization.embeddings import EventEmbeddingManager
from src.vectorization.build_index import FaissIndexBuilder


class TestEventEmbeddingManager:
    """Tests pour le gestionnaire d'embeddings."""

    @pytest.fixture
    def manager(self):
        """Fixture pour créer un manager."""
        return EventEmbeddingManager()

    @pytest.fixture
    def sample_event(self):
        """Fixture avec un événement d'exemple."""
        return {
            "id": "evt_001",
            "title": "Concert de Jazz",
            "description": "Un magnifique concert avec les meilleurs musiciens de jazz",
            "location": "Toulouse",
        }

    def test_manager_initialization(self, manager):
        """Tester l'initialisation du manager."""
        assert manager is not None
        assert manager.model_dimension == 384

    def test_embed_text(self, manager):
        """Tester la création d'un embedding pour un texte."""
        text = "Concert de jazz à Toulouse"
        embedding = manager.embed_text(text)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_empty_text(self, manager):
        """Tester que le texte vide retourne None."""
        embedding = manager.embed_text("")
        assert embedding is None

    def test_embed_event(self, manager, sample_event):
        """Tester la vectorisation d'un événement."""
        events = [sample_event]
        embeddings, valid_events = manager.embed_events(events)
        
        assert len(embeddings) == 1
        assert len(valid_events) == 1
        assert valid_events[0]["title"] == "Concert de Jazz"

    def test_embed_multiple_events(self, manager):
        """Tester la vectorisation de plusieurs événements."""
        events = [
            {
                "id": "evt_001",
                "title": "Concert de Jazz",
                "description": "Un magnifique concert",
            },
            {
                "id": "evt_002",
                "title": "Exposition d'Art",
                "description": "Une exposition remarquable",
            },
        ]
        
        embeddings, valid_events = manager.embed_events(events)
        
        assert len(embeddings) == 2
        assert len(valid_events) == 2

    def test_embedding_similarity(self, manager):
        """Tester que les embeddings similaires ont une distance faible."""
        text1 = "Concert de jazz à Toulouse"
        text2 = "Concert musical à Toulouse"  # Similaire
        text3 = "Exposition d'art à Paris"    # Différent
        
        emb1 = manager.embed_text(text1)
        emb2 = manager.embed_text(text2)
        emb3 = manager.embed_text(text3)
        
        # Calculer les distances
        dist_1_2 = np.linalg.norm(emb1 - emb2)
        dist_1_3 = np.linalg.norm(emb1 - emb3)
        
        # Les textes similaires doivent être plus proches
        assert dist_1_2 < dist_1_3

    def test_get_dimension(self, manager):
        """Tester que la dimension est correcte."""
        assert manager.get_dimension() == 384


class TestFaissIndexBuilder:
    """Tests pour le builder d'index Faiss."""

    @pytest.fixture
    def builder(self):
        """Fixture pour créer un builder."""
        return FaissIndexBuilder()

    @pytest.fixture
    def sample_events(self):
        """Fixture avec des événements d'exemple."""
        return [
            {
                "id": "evt_001",
                "title": "Concert de Jazz",
                "description": "Un magnifique concert avec les meilleurs musiciens",
                "location": "Toulouse",
            },
            {
                "id": "evt_002",
                "title": "Exposition d'Art Moderne",
                "description": "Une exposition remarquable d'art contemporain",
                "location": "Toulouse",
            },
            {
                "id": "evt_003",
                "title": "Festival de Théâtre",
                "description": "Le plus grand festival de théâtre du sud",
                "location": "Toulouse",
            },
            {
                "id": "evt_004",
                "title": "Conférence sur l'Histoire",
                "description": "Une conférence fascinante sur 2000 ans d'histoire",
                "location": "Toulouse",
            },
        ]

    def test_builder_initialization(self, builder):
        """Tester l'initialisation du builder."""
        assert builder is not None
        assert builder.index is None

    def test_build_index(self, builder, sample_events):
        """Tester la construction d'un index."""
        index, valid_events = builder.build_index(sample_events)
        
        assert index is not None
        assert len(valid_events) == 4
        assert index.ntotal == 4

    def test_build_index_empty(self, builder):
        """Tester la construction avec des événements vides."""
        invalid_events = [
            {"id": "evt_001"},  # Sans titre ni description
            {"id": "evt_002"},
        ]
        
        index, valid_events = builder.build_index(invalid_events)
        
        assert index is None or index.ntotal == 0
        assert len(valid_events) == 0

    def test_search_similar_events(self, builder, sample_events):
        """Tester la recherche d'événements similaires."""
        index, valid_events = builder.build_index(sample_events)
        
        # Créer un embedding pour une requête similaire à "Jazz"
        manager = EventEmbeddingManager()
        query = manager.embed_text("Concert de musique jazz")
        
        # Chercher les 2 événements les plus proches
        results = builder.search(query, k=2)
        
        assert len(results) <= 2
        assert len(results) > 0

    def test_search_returns_events(self, builder, sample_events):
        """Tester que la recherche retourne des événements avec métadonnées."""
        index, valid_events = builder.build_index(sample_events)
        
        manager = EventEmbeddingManager()
        query = manager.embed_text("Théâtre")
        
        results = builder.search(query, k=1)
        
        assert len(results) >= 1
        assert "id" in results[0]
        assert "title" in results[0]
        assert "description" in results[0]

    def test_save_and_load_index(self, builder, sample_events):
        """Tester la sauvegarde et le chargement d'un index."""
        # Construire l'index
        index, valid_events = builder.build_index(sample_events)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Sauvegarder
            builder.save_index(tmpdir)
            
            # Vérifier que les fichiers existent
            assert (tmpdir / "faiss_index.bin").exists()
            assert (tmpdir / "events_metadata.json").exists()
            assert (tmpdir / "index_metadata.pkl").exists()
            
            # Charger
            loaded_index, loaded_events, metadata = FaissIndexBuilder.load_index(tmpdir)
            
            assert loaded_index is not None
            assert len(loaded_events) == 4
            assert metadata["num_events"] == 4
            assert metadata["embedding_dimension"] == 384

    def test_index_persistence(self, builder, sample_events):
        """Tester que l'index persiste correctement."""
        index, valid_events = builder.build_index(sample_events)
        num_vectors_before = index.ntotal
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            builder.save_index(tmpdir)
            
            loaded_index, loaded_events, metadata = FaissIndexBuilder.load_index(tmpdir)
            
            # Vérifier que les dimensions sont préservées
            assert loaded_index.ntotal == num_vectors_before
            assert len(loaded_events) == num_vectors_before


class TestVectorizationIntegration:
    """Tests d'intégration pour la vectorisation complète."""

    def test_end_to_end_vectorization(self):
        """Tester le pipeline complet de vectorisation."""
        # Créer des événements d'exemple
        events = [
            {
                "id": "jazz_001",
                "title": "Soirée Jazz Toulouse",
                "description": "Une soirée exceptionnelle de jazz live avec les plus grands artistes",
            },
            {
                "id": "art_001",
                "title": "Exposition d'Art Contemporain",
                "description": "Découvrez les œuvres d'art modernes et contemporaines",
            },
        ]
        
        # Construire l'index
        builder = FaissIndexBuilder()
        index, valid_events = builder.build_index(events)
        
        # Chercher
        manager = EventEmbeddingManager()
        query = manager.embed_text("Concert jazz")
        results = builder.search(query, k=1)
        
        # Vérifier que le premier résultat est un événement jazz
        assert len(results) > 0
        assert "jazz" in results[0]["title"].lower()

    def test_vectorization_consistency(self):
        """Tester que les embeddings sont cohérents."""
        manager = EventEmbeddingManager()
        
        text = "Concert de jazz à Toulouse"
        emb1 = manager.embed_text(text)
        emb2 = manager.embed_text(text)
        
        # Les embeddings du même texte doivent être identiques
        assert np.allclose(emb1, emb2, rtol=1e-5)
