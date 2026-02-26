"""
Script pour récupérer les événements culturels depuis l'API Open Agenda.
Focus sur la ville de Toulouse.
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de l'API Open Agenda
OPENAGENDA_API_BASE_URL = "https://api.openagenda.com/v2"
OPENAGENDA_API_KEY = "123123"  # Clé publique par défaut pour les tests

# Configuration de Toulouse
TOULOUSE_LOCATION = {
    "city": "Toulouse",
    "country": "FR",
    "latitude": 43.6047,
    "longitude": 1.4442,
}

# Rayon de recherche en km (pour capturer les événements à Toulouse)
SEARCH_RADIUS_KM = 20

# Plages de dates
def get_date_range():
    """Retourne les dates pour les 12 derniers mois + événements à venir."""
    end_date = datetime.now() + timedelta(days=365)
    start_date = datetime.now() - timedelta(days=365)
    return start_date, end_date


class OpenAgendaEventFetcher:
    """Classe pour récupérer les événements depuis l'API Open Agenda."""

    def __init__(self, api_key: str = OPENAGENDA_API_KEY):
        """
        Initialiser le fetcher.
        
        Args:
            api_key: Clé API Open Agenda
        """
        self.api_key = api_key
        self.base_url = OPENAGENDA_API_BASE_URL
        self.session = requests.Session()

    def fetch_events(
        self,
        city: str = "Toulouse",
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict]:
        """
        Récupérer les événements depuis l'API.
        
        Args:
            city: Nom de la ville
            limit: Nombre max d'événements à récupérer
            offset: Offset pour la pagination
            
        Returns:
            Liste des événements
        """
        try:
            # Construire l'URL avec les paramètres
            url = f"{self.base_url}/agendas"
            
            params = {
                "apiKey": self.api_key,
                "search": city,
                "limit": limit,
                "offset": offset,
                "include": "events",
            }
            
            logger.info(f"Fetching events for {city} from Open Agenda API...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extraire les événements de la réponse
            events = []
            if "data" in data:
                for agenda in data["data"]:
                    if "events" in agenda:
                        events.extend(agenda["events"])
            
            logger.info(f"Successfully fetched {len(events)} events")
            return events
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching events from API: {e}")
            return []

    def filter_events(self, events: List[Dict]) -> List[Dict]:
        """
        Filtrer les événements selon des critères.
        
        Args:
            events: Liste des événements
            
        Returns:
            Liste des événements filtrés
        """
        filtered = []
        start_date, end_date = get_date_range()
        
        for event in events:
            try:
                # Vérifier que l'événement a les champs requis
                if not event.get("title"):
                    continue
                    
                # Filtrer par date si disponible
                if "date" in event and event["date"]:
                    event_date = datetime.fromisoformat(event["date"]["start"].replace("Z", "+00:00"))
                    if not (start_date <= event_date <= end_date):
                        continue
                
                filtered.append(event)
                
            except Exception as e:
                logger.warning(f"Error filtering event: {e}")
                continue
        
        logger.info(f"Filtered to {len(filtered)} events")
        return filtered


def save_events_to_json(events: List[Dict], output_path: Path) -> None:
    """
    Sauvegarder les événements en JSON.
    
    Args:
        events: Liste des événements
        output_path: Chemin du fichier de sortie
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Events saved to {output_path}")


def main():
    """Script principal pour récupérer les événements."""
    
    # Initialiser le fetcher
    fetcher = OpenAgendaEventFetcher()
    
    # Récupérer les événements de Toulouse
    events = fetcher.fetch_events(city=TOULOUSE_LOCATION["city"])
    
    # Filtrer les événements
    filtered_events = fetcher.filter_events(events)
    
    # Sauvegarder en JSON
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    output_path = output_dir / "toulouse_events_raw.json"
    
    save_events_to_json(filtered_events, output_path)
    
    logger.info(f"✅ Successfully fetched and saved {len(filtered_events)} events")
    print(f"\n✅ {len(filtered_events)} events saved to {output_path}")


if __name__ == "__main__":
    main()
