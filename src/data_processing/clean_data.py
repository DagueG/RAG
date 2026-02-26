"""
Script pour nettoyer et préprocesser les données d'événements.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventDataCleaner:
    """Classe pour nettoyer et préprocesser les données d'événements."""

    def __init__(self):
        """Initialiser le cleaner."""
        self.required_fields = ["title", "description"]
        self.optional_fields = ["date", "location", "uid", "image"]

    def load_raw_events(self, input_path: Path) -> List[Dict]:
        """
        Charger les événements bruts depuis un fichier JSON.
        
        Args:
            input_path: Chemin du fichier JSON
            
        Returns:
            Liste des événements
        """
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            return []
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                events = json.load(f)
            logger.info(f"Loaded {len(events)} raw events from {input_path}")
            return events
        except Exception as e:
            logger.error(f"Error loading events: {e}")
            return []

    def clean_event(self, event: Dict) -> Optional[Dict]:
        """
        Nettoyer un seul événement.
        
        Args:
            event: Événement brut
            
        Returns:
            Événement nettoyé ou None si invalid
        """
        try:
            # Vérifier les champs obligatoires
            if not event.get("title"):
                return None
            
            # Créer l'événement nettoyé
            cleaned = {
                "id": event.get("uid", ""),
                "title": str(event.get("title", "")).strip(),
                "description": str(event.get("description", "")).strip(),
                "date_start": self._extract_date(event.get("date", {}).get("start")) if event.get("date") else None,
                "date_end": self._extract_date(event.get("date", {}).get("end")) if event.get("date") else None,
                "location": self._extract_location(event),
                "image_url": event.get("image", {}).get("url") if event.get("image") else None,
                "url": event.get("url", ""),
                "source": "openagenda",
            }
            
            # Vérifier qu'on a au moins un titre et une description
            if not cleaned["title"] or not cleaned["description"]:
                return None
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error cleaning event: {e}")
            return None

    @staticmethod
    def _extract_date(date_str: Optional[str]) -> Optional[str]:
        """Extraire et formater une date."""
        if not date_str:
            return None
        try:
            # Essayer de parser différents formats
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.isoformat()
        except Exception:
            return date_str

    @staticmethod
    def _extract_location(event: Dict) -> str:
        """Extraire la localisation d'un événement."""
        location_parts = []
        
        if event.get("location"):
            location = event["location"]
            if isinstance(location, dict):
                if location.get("address"):
                    location_parts.append(location["address"])
                if location.get("city"):
                    location_parts.append(location["city"])
                if location.get("region"):
                    location_parts.append(location["region"])
            else:
                location_parts.append(str(location))
        
        return ", ".join(location_parts) if location_parts else "Unknown location"

    def clean_events(self, events: List[Dict]) -> List[Dict]:
        """
        Nettoyer une liste d'événements.
        
        Args:
            events: Liste des événements bruts
            
        Returns:
            Liste des événements nettoyés
        """
        cleaned = []
        for event in events:
            cleaned_event = self.clean_event(event)
            if cleaned_event:
                cleaned.append(cleaned_event)
        
        logger.info(f"Cleaned {len(cleaned)} out of {len(events)} events")
        return cleaned

    def save_to_csv(self, events: List[Dict], output_path: Path) -> None:
        """
        Sauvegarder les événements en CSV.
        
        Args:
            events: Liste des événements
            output_path: Chemin du fichier de sortie
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(events)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Events saved to CSV: {output_path}")

    def save_to_json(self, events: List[Dict], output_path: Path) -> None:
        """
        Sauvegarder les événements en JSON.
        
        Args:
            events: Liste des événements
            output_path: Chemin du fichier de sortie
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        logger.info(f"Events saved to JSON: {output_path}")

    def get_statistics(self, events: List[Dict]) -> Dict:
        """
        Obtenir des statistiques sur les événements.
        
        Args:
            events: Liste des événements
            
        Returns:
            Dictionnaire de statistiques
        """
        df = pd.DataFrame(events)
        
        stats = {
            "total_events": len(events),
            "events_with_date": df["date_start"].notna().sum(),
            "events_with_location": df["location"].notna().sum(),
            "events_with_image": df["image_url"].notna().sum(),
            "avg_title_length": df["title"].str.len().mean(),
            "avg_description_length": df["description"].str.len().mean(),
        }
        
        return stats


def main():
    """Script principal pour nettoyer les données."""
    
    # Chemins
    base_dir = Path(__file__).parent.parent.parent
    raw_input = base_dir / "data" / "raw" / "toulouse_events_raw.json"
    processed_output_csv = base_dir / "data" / "processed" / "toulouse_events.csv"
    processed_output_json = base_dir / "data" / "processed" / "toulouse_events.json"
    
    # Initialiser le cleaner
    cleaner = EventDataCleaner()
    
    # Charger les événements bruts
    raw_events = cleaner.load_raw_events(raw_input)
    
    if not raw_events:
        logger.warning("No raw events found. Please run fetch_events.py first.")
        print("❌ No raw events found. Please fetch data first with fetch_events.py")
        return
    
    # Nettoyer les événements
    cleaned_events = cleaner.clean_events(raw_events)
    
    # Sauvegarder les résultats
    cleaner.save_to_csv(cleaned_events, processed_output_csv)
    cleaner.save_to_json(cleaned_events, processed_output_json)
    
    # Afficher les statistiques
    stats = cleaner.get_statistics(cleaned_events)
    
    print(f"\n📊 Data Cleaning Statistics:")
    print(f"   Total events: {stats['total_events']}")
    print(f"   Events with date: {stats['events_with_date']}")
    print(f"   Events with location: {stats['events_with_location']}")
    print(f"   Events with image: {stats['events_with_image']}")
    print(f"   Avg title length: {stats['avg_title_length']:.1f} chars")
    print(f"   Avg description length: {stats['avg_description_length']:.1f} chars")
    print(f"\n✅ Data cleaning complete!")
    print(f"   CSV: {processed_output_csv}")
    print(f"   JSON: {processed_output_json}")


if __name__ == "__main__":
    main()
