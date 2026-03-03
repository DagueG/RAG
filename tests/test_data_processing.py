"""
Tests unitaires pour le preprocessing des données.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.data_processing.clean_data import EventDataCleaner


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

class TestEventDataQuality:
    """Tests pour valider la qualité des données d'événements récupérées."""

    @pytest.fixture
    def fetcher(self):
        """Fixture pour créer un fetcher."""
        from src.data_processing.fetch_events import OpenAgendaEventFetcher
        return OpenAgendaEventFetcher()

    def test_filter_events_rejects_missing_title(self, fetcher):
        """Tester que les événements sans titre sont rejetés."""
        event_no_title = {
            "title_fr": None,
            "description_fr": "Event without title",
            "firstdate_begin": "2026-03-15T19:00:00Z",
        }
        
        filtered = fetcher.filter_events([event_no_title])
        assert len(filtered) == 0, "Event without title should be rejected"

    def test_filter_events_rejects_missing_date(self, fetcher):
        """Tester que les événements sans date sont rejetés."""
        event_no_date = {
            "title_fr": "Event without date",
            "description_fr": "A concert",
            "firstdate_begin": None,
        }
        
        filtered = fetcher.filter_events([event_no_date])
        assert len(filtered) == 0, "Event without date should be rejected"

    def test_filter_events_rejects_missing_description(self, fetcher):
        """Tester que les événements sans description sont rejetés."""
        event_no_desc = {
            "title_fr": "Event without description",
            "description_fr": None,
            "longdescription_fr": None,
            "firstdate_begin": "2026-03-15T19:00:00Z",
        }
        
        filtered = fetcher.filter_events([event_no_desc])
        assert len(filtered) == 0, "Event without description should be rejected"

    def test_filter_events_accepts_valid_event(self, fetcher):
        """Tester qu'un événement valide est accepté."""
        valid_event = {
            "title_fr": "Exposition d'Art Moderne",
            "description_fr": "Une magnifique exposition d'art contemporain",
            "firstdate_begin": "2026-06-15T10:00:00Z",
            "location_address": "Musée d'Art, Toulouse",
        }
        
        filtered = fetcher.filter_events([valid_event])
        assert len(filtered) == 1, "Valid event should be accepted"
        assert filtered[0]["title_fr"] == "Exposition d'Art Moderne"

    def test_filter_events_accepts_with_longdescription(self, fetcher):
        """Tester qu'un événement avec longdescription_fr est accepté."""
        valid_event = {
            "title_fr": "Concert de Jazz",
            "description_fr": None,
            "longdescription_fr": "Un super concert de jazz avec des artistes internationaux",
            "firstdate_begin": "2026-05-20T20:00:00Z",
        }
        
        filtered = fetcher.filter_events([valid_event])
        assert len(filtered) == 1, "Event with longdescription_fr should be accepted"

    def test_filter_events_rejects_invalid_date_format(self, fetcher):
        """Tester que les événements avec formats de date invalides sont rejetés."""
        event_bad_date = {
            "title_fr": "Event with bad date",
            "description_fr": "Some description",
            "firstdate_begin": "not-a-date-format",
        }
        
        filtered = fetcher.filter_events([event_bad_date])
        assert len(filtered) == 0, "Event with invalid date format should be rejected"

    def test_filter_events_accepts_mixed_batch(self, fetcher):
        """Tester le filtrage d'un lot mixte d'événements valides et invalides."""
        events = [
            # Valide
            {
                "title_fr": "Concert 1",
                "description_fr": "Valid concert",
                "firstdate_begin": "2026-03-15T19:00:00Z",
            },
            # Invalide - pas de titre
            {
                "title_fr": None,
                "description_fr": "No title",
                "firstdate_begin": "2026-03-20T19:00:00Z",
            },
            # Valide
            {
                "title_fr": "Exposition 2",
                "description_fr": "Art exhibition",
                "firstdate_begin": "2026-04-10T10:00:00Z",
            },
            # Invalide - pas de description
            {
                "title_fr": "No Description Event",
                "description_fr": None,
                "longdescription_fr": None,
                "firstdate_begin": "2026-06-01T18:00:00Z",
            },
        ]
        
        filtered = fetcher.filter_events(events)
        assert len(filtered) == 2, "Should accept 2 valid events and reject 2 invalid"
        assert filtered[0]["title_fr"] == "Concert 1"
        assert filtered[1]["title_fr"] == "Exposition 2"

class TestFetchedEventsValidation:
    """Tests pour valider les données réellement récupérées depuis l'API."""

    def test_all_events_have_future_dates(self):
        """Vérifier que TOUS les événements ont des dates futures (>= aujourd'hui)."""
        from pathlib import Path
        from datetime import datetime, timezone
        
        raw_file = Path(__file__).parent.parent / "data" / "raw" / "toulouse_events_raw.json"
        
        if not raw_file.exists():
            pytest.skip("Raw events file not found - run fetch_events.py first")
        
        with open(raw_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
        
        assert len(events) > 0, "Should have fetched at least 1 event"
        
        now = datetime.now(timezone.utc)
        
        for event in events:
            event_date_str = event.get("firstdate_begin")
            assert event_date_str is not None, f"Event {event.get('title_fr')} missing date"
            
            event_date = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
            assert event_date >= now, f"Event {event.get('title_fr')} date {event_date_str} is in the past"

    def test_all_events_within_1_year(self):
        """Vérifier que TOUS les événements sont à moins d'1 an."""
        from pathlib import Path
        from datetime import datetime, timedelta, timezone
        
        raw_file = Path(__file__).parent.parent / "data" / "raw" / "toulouse_events_raw.json"
        
        if not raw_file.exists():
            pytest.skip("Raw events file not found - run fetch_events.py first")
        
        with open(raw_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
        
        assert len(events) > 0, "Should have fetched at least 1 event"
        
        now = datetime.now(timezone.utc)
        one_year_later = now + timedelta(days=365)
        
        for event in events:
            event_date_str = event.get("firstdate_begin")
            event_date = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
            assert event_date <= one_year_later, f"Event {event.get('title_fr')} date {event_date_str} is more than 1 year away"

    def test_all_events_from_toulouse(self):
        """Vérifier que les événements sont de Toulouse."""
        from pathlib import Path
        
        raw_file = Path(__file__).parent.parent / "data" / "raw" / "toulouse_events_raw.json"
        
        if not raw_file.exists():
            pytest.skip("Raw events file not found - run fetch_events.py first")
        
        with open(raw_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
        
        assert len(events) > 0, "Should have fetched at least 1 event"
        
        for event in events:
            city = event.get("location_city", "").lower()
            assert "toulouse" in city, f"Event {event.get('title_fr')} is not from Toulouse (city: {city})"

    def test_events_have_required_fields(self):
        """Vérifier que tous les événements ont les champs obligatoires."""
        from pathlib import Path
        
        raw_file = Path(__file__).parent.parent / "data" / "raw" / "toulouse_events_raw.json"
        
        if not raw_file.exists():
            pytest.skip("Raw events file not found - run fetch_events.py first")
        
        with open(raw_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
        
        assert len(events) > 0, "Should have fetched at least 1 event"
        
        required_fields = ["title_fr", "firstdate_begin"]
        
        for i, event in enumerate(events):
            for field in required_fields:
                assert field in event, f"Event {i} ({event.get('title_fr')}) missing required field: {field}"
                assert event[field], f"Event {i} ({event.get('title_fr')}) has empty {field}"