"""
Tests unitaires pour le preprocessing des données.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.data_processing.clean_data import EventDataCleaner
from src.data_processing.fetch_events import OpenAgendaEventFetcher


class TestEventDataCleaner:
    """Tests pour la classe EventDataCleaner."""

    @pytest.fixture
    def cleaner(self):
        """Fixture pour créer un cleaner."""
        return EventDataCleaner()

    @pytest.fixture
    def sample_raw_event(self):
        """Fixture avec un événement brut d'exemple."""
        return {
            "uid": "event_123",
            "title": "Concert de Jazz à Toulouse",
            "description": "Un magnifique concert de jazz avec les meilleurs musiciens",
            "date": {
                "start": "2024-03-15T19:00:00Z",
                "end": "2024-03-15T22:00:00Z",
            },
            "location": {
                "address": "123 Rue de la Paix",
                "city": "Toulouse",
                "region": "Occitanie",
            },
            "image": {
                "url": "https://example.com/image.jpg"
            },
            "url": "https://example.com/event",
        }

    def test_clean_event_valid(self, cleaner, sample_raw_event):
        """Tester le nettoyage d'un événement valide."""
        cleaned = cleaner.clean_event(sample_raw_event)
        
        assert cleaned is not None
        assert cleaned["title"] == "Concert de Jazz à Toulouse"
        assert "jazz" in cleaned["description"].lower()
        assert cleaned["id"] == "event_123"
        assert cleaned["source"] == "openagenda"

    def test_clean_event_missing_title(self, cleaner, sample_raw_event):
        """Tester qu'un événement sans titre est rejeté."""
        sample_raw_event["title"] = ""
        cleaned = cleaner.clean_event(sample_raw_event)
        assert cleaned is None

    def test_clean_event_missing_description(self, cleaner, sample_raw_event):
        """Tester qu'un événement sans description est rejeté."""
        sample_raw_event["description"] = ""
        cleaned = cleaner.clean_event(sample_raw_event)
        assert cleaned is None

    def test_extract_location(self, cleaner):
        """Tester l'extraction de localisation."""
        event = {
            "location": {
                "address": "123 Rue de la Paix",
                "city": "Toulouse",
                "region": "Occitanie",
            }
        }
        location = cleaner._extract_location(event)
        assert "Toulouse" in location
        assert "Occitanie" in location

    def test_extract_date(self, cleaner):
        """Tester l'extraction de date."""
        date_str = "2024-03-15T19:00:00Z"
        result = cleaner._extract_date(date_str)
        assert result is not None
        assert "2024" in result

    def test_clean_events_list(self, cleaner, sample_raw_event):
        """Tester le nettoyage d'une liste d'événements."""
        events = [sample_raw_event, sample_raw_event.copy()]
        cleaned = cleaner.clean_events(events)
        
        assert len(cleaned) == 2
        assert all("id" in e for e in cleaned)

    def test_save_to_csv(self, cleaner, sample_raw_event):
        """Tester la sauvegarde en CSV."""
        cleaned_events = [cleaner.clean_event(sample_raw_event)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "events.csv"
            cleaner.save_to_csv(cleaned_events, output_path)
            
            assert output_path.exists()
            # Vérifier que le fichier contient les données
            with open(output_path, "r") as f:
                content = f.read()
                assert "title" in content
                assert "Concert de Jazz" in content

    def test_save_to_json(self, cleaner, sample_raw_event):
        """Tester la sauvegarde en JSON."""
        cleaned_events = [cleaner.clean_event(sample_raw_event)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "events.json"
            cleaner.save_to_json(cleaned_events, output_path)
            
            assert output_path.exists()
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]["title"] == "Concert de Jazz à Toulouse"

    def test_get_statistics(self, cleaner, sample_raw_event):
        """Tester le calcul des statistiques."""
        cleaned_events = [cleaner.clean_event(sample_raw_event)]
        stats = cleaner.get_statistics(cleaned_events)
        
        assert stats["total_events"] == 1
        assert stats["events_with_date"] == 1
        assert stats["avg_title_length"] > 0

    def test_load_raw_events_file_not_found(self, cleaner):
        """Tester le chargement d'un fichier inexistant."""
        events = cleaner.load_raw_events(Path("nonexistent.json"))
        assert events == []


class TestOpenAgendaEventFetcher:
    """Tests pour la classe OpenAgendaEventFetcher."""

    @pytest.fixture
    def fetcher(self):
        """Fixture pour créer un fetcher."""
        return OpenAgendaEventFetcher()

    def test_fetcher_initialization(self, fetcher):
        """Tester l'initialisation du fetcher."""
        assert fetcher.api_key == "123123"
        assert fetcher.base_url == "https://api.openagenda.com/v2"
        assert fetcher.session is not None

    def test_filter_events_empty_list(self, fetcher):
        """Tester le filtrage d'une liste vide."""
        filtered = fetcher.filter_events([])
        assert filtered == []

    def test_filter_events_missing_title(self, fetcher):
        """Tester que les événements sans titre sont filtrés."""
        events = [
            {"description": "Event description"},  # Pas de titre
        ]
        filtered = fetcher.filter_events(events)
        assert len(filtered) == 0


class TestDataIntegration:
    """Tests d'intégration pour le pipeline complet."""

    def test_cleaner_with_real_raw_format(self):
        """Tester le cleaner avec un format réaliste."""
        cleaner = EventDataCleaner()
        
        # Événement au format réaliste
        raw_sample = {
            "uid": "test_123",
            "title": "Festival d'Art Contemporain",
            "description": "Un événement culturel majeur à Toulouse",
            "date": {
                "start": "2024-06-01T10:00:00Z",
                "end": "2024-06-01T18:00:00Z",
            },
            "location": {
                "address": "Centre Culturel",
                "city": "Toulouse",
            },
            "image": {"url": "https://example.com/image.jpg"},
            "url": "https://example.com",
        }
        
        cleaned = cleaner.clean_event(raw_sample)
        
        # Vérifications
        assert cleaned is not None
        assert cleaned["title"] == "Festival d'Art Contemporain"
        assert "contemporain" in cleaned["title"].lower()
        assert cleaned["date_start"] is not None
        assert "Toulouse" in cleaned["location"]
        assert cleaned["source"] == "openagenda"
