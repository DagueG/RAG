"""
Script pour récupérer les événements culturels depuis l'API OpenDataSoft.
Focus sur la ville de Toulouse.
Utilise les données publiques d'OpenAgenda exposées par OpenDataSoft.
"""

import requests
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de l'API OpenDataSoft (gratuit, pas de clé requise)
OPENDATASOFT_API_BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

# Configuration de Toulouse
TOULOUSE_LOCATION = {
    "city": "Toulouse",
    "country": "FR",
    "latitude": 43.6047,
    "longitude": 1.4442,
}

# Plages de dates
def get_date_range():
    """Retourne les dates pour les événements récents (actuels et futurs, < 1 an)."""
    # Utiliser timezone aware pour comparaison avec les dates de l'API
    start_date = datetime.now(timezone.utc)  # Événements actuels et futurs seulement
    end_date = datetime.now(timezone.utc) + timedelta(days=365)  # Jusqu'à 1 an dans le futur
    return start_date, end_date


class OpenAgendaEventFetcher:
    """Classe pour récupérer les événements depuis l'API OpenDataSoft."""

    def __init__(self):
        """Initialiser le fetcher."""
        self.base_url = OPENDATASOFT_API_BASE_URL
        self.session = requests.Session()

    def fetch_events(
        self,
        city: str = "Toulouse",
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Récupérer les événements depuis l'API OpenDataSoft.
        Filtre DIRECTEMENT À L'API: ville + dates (futur, < 1 an)
        
        Args:
            city: Nom de la ville
            limit: Nombre max d'événements à récupérer (max 100 par requête)
            offset: Offset pour la pagination
            
        Returns:
            Liste des événements (déjà filtrés par l'API)
        """
        try:
            # OpenDataSoft API - limit max is 100
            limit = min(limit, 100)
            
            # Construire la plage de dates pour le filtre API
            start_date, end_date = get_date_range()
            start_iso = start_date.isoformat().split('+')[0] + "Z"  # Format ISO compatible
            end_iso = end_date.isoformat().split('+')[0] + "Z"
            
            # Construire les paramètres de requête avec filtres
            params = {
                "limit": limit,
                "offset": offset,
                "refine": f"location_city:{city}",  # Filtre ville
                "where": f"firstdate_begin>=\"{start_iso}\" AND firstdate_begin<=\"{end_iso}\""  # Filtre dates (futur, < 1 an)
            }
            
            logger.info(f"Fetching events for {city} from OpenDataSoft API (future events only, < 1 year)...")
            logger.info(f"Date range: {start_iso} to {end_iso}")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extraire les événements de la réponse (format OpenDataSoft)
            events = data.get("results", [])
            
            logger.info(f"Successfully fetched {len(events)} events")
            return events
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching events from API: {e}")
            return []

    def filter_events(self, events: List[Dict]) -> List[Dict]:
        """
        Valider la structure des événements (données complètes).
        NOTE: Les dates sont déjà filtrées à l'API, on valide juste les champs obligatoires.
        
        Args:
            events: Liste des événements (pré-filtrés par l'API)
            
        Returns:
            Liste des événements valides
        """
        filtered = []
        rejected = []
        
        for event in events:
            try:
                event_title = event.get("title_fr", "Unknown")
                
                # Vérifier que l'événement a les champs requis (titre français)
                if not event.get("title_fr"):
                    rejected.append((event_title, "Missing title_fr"))
                    continue
                    
                # Vérifier qu'il y a une description
                description = event.get("description_fr") or event.get("longdescription_fr", "")
                if not description:
                    rejected.append((event_title, "Missing description"))
                    continue
                
                # Note: Date est déjà validée par l'API (via where clause)
                if not event.get("firstdate_begin"):
                    rejected.append((event_title, "Missing event date (firstdate_begin)"))
                    continue
                
                # Valider le format de la date
                try:
                    datetime.fromisoformat(event["firstdate_begin"].replace("Z", "+00:00"))
                except (ValueError, TypeError) as e:
                    rejected.append((event_title, f"Invalid date format: {event.get('firstdate_begin')}"))
                    continue
                
                filtered.append(event)
                
            except Exception as e:
                logger.warning(f"Error validating event: {e}")
                rejected.append((event.get("title_fr", "Unknown"), str(e)))
                continue
        
        logger.info(f"Validated events: {len(filtered)} accepted, {len(rejected)} rejected")
        
        # Log rejections if any
        if rejected:
            logger.warning(f"Rejected {len(rejected)} events:")
            for title, reason in rejected[:5]:  # Show first 5 rejections
                logger.warning(f"  - {title}: {reason}")
            if len(rejected) > 5:
                logger.warning(f"  ... and {len(rejected) - 5} more")
        
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
