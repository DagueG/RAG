"""
Script de démonstration du pipeline de data processing.
Crée des données d'exemple et les traite.
"""

import json
from pathlib import Path
from src.data_processing.clean_data import EventDataCleaner


def create_sample_data():
    """Créer des données d'exemple pour tester le pipeline."""
    
    sample_events = [
        {
            "uid": "evt_001",
            "title": "Concert de Jazz au Théâtre du Capitole",
            "description": "Un spectaculaire concert de jazz avec les meilleures formations de Toulouse. Musique live en direct.",
            "date": {
                "start": "2024-03-15T19:00:00Z",
                "end": "2024-03-15T22:00:00Z",
            },
            "location": {
                "address": "Place du Capitole",
                "city": "Toulouse",
                "region": "Occitanie",
            },
            "image": {
                "url": "https://example.com/jazz.jpg"
            },
            "url": "https://example.com/events/jazz",
        },
        {
            "uid": "evt_002",
            "title": "Exposition d'Art Moderne au Musée d'Art Moderne",
            "description": "Une exposition remarquable présentant les œuvres des plus grands artistes modernes. À ne pas manquer!",
            "date": {
                "start": "2024-03-20T09:00:00Z",
                "end": "2024-06-20T18:00:00Z",
            },
            "location": {
                "address": "76 Allee Charles de Fitte",
                "city": "Toulouse",
                "region": "Occitanie",
            },
            "image": {
                "url": "https://example.com/art.jpg"
            },
            "url": "https://example.com/events/art",
        },
        {
            "uid": "evt_003",
            "title": "Festival de Théâtre de Toulouse",
            "description": "Le plus grand festival de théâtre du sud. Des spectacles de qualité mondiale, des pièces classiques et modernes.",
            "date": {
                "start": "2024-05-01T20:00:00Z",
                "end": "2024-05-30T23:00:00Z",
            },
            "location": {
                "address": "Multiple venues",
                "city": "Toulouse",
                "region": "Occitanie",
            },
            "image": {
                "url": "https://example.com/theatre.jpg"
            },
            "url": "https://example.com/events/theatre",
        },
        {
            "uid": "evt_004",
            "title": "Conférence: L'Histoire de Toulouse",
            "description": "Une conférence fascinante sur 2000 ans d'histoire toulousaine. Animée par des historiens reconnus.",
            "date": {
                "start": "2024-04-10T18:30:00Z",
                "end": "2024-04-10T20:00:00Z",
            },
            "location": {
                "address": "Bibliothèque Municipale",
                "city": "Toulouse",
                "region": "Occitanie",
            },
            "image": {
                "url": "https://example.com/conference.jpg"
            },
            "url": "https://example.com/events/conference",
        },
        {
            "uid": "evt_005",
            "title": "Festival de Musique Classique",
            "description": "Une sélection exclusive de concerts de musique classique avec les meilleures orchestres nationales et internationales.",
            "date": {
                "start": "2024-05-15T19:30:00Z",
                "end": "2024-05-15T21:30:00Z",
            },
            "location": {
                "address": "Salle de Concert du Capitole",
                "city": "Toulouse",
                "region": "Occitanie",
            },
            "image": {
                "url": "https://example.com/classique.jpg"
            },
            "url": "https://example.com/events/classique",
        },
    ]
    
    return sample_events


def main():
    """Exécuter la démonstration du pipeline."""
    
    print("=" * 60)
    print("DEMO: Data Processing Pipeline for Toulouse Events")
    print("=" * 60)
    
    # Créer les répertoires nécessaires
    # Depuis tests/ on remonte à la racine avec .parent
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer les données d'exemple
    print("\n[1] Creating sample data...")
    sample_data = create_sample_data()
    raw_file = raw_dir / "toulouse_events_demo.json"
    
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"   OK Created {len(sample_data)} sample events in {raw_file}")
    
    # Nettoyer les données
    print("\n[2] Cleaning and preprocessing data...")
    cleaner = EventDataCleaner()
    cleaned_events = cleaner.clean_events(sample_data)
    
    print(f"   OK Cleaned {len(cleaned_events)} events")
    
    # Sauvegarder en CSV
    csv_file = processed_dir / "toulouse_events_demo.csv"
    cleaner.save_to_csv(cleaned_events, csv_file)
    print(f"   OK Saved to CSV: {csv_file}")
    
    # Sauvegarder en JSON
    json_file = processed_dir / "toulouse_events_demo.json"
    cleaner.save_to_json(cleaned_events, json_file)
    print(f"   OK Saved to JSON: {json_file}")
    
    # Afficher les statistiques
    print("\n[3] Data Statistics:")
    stats = cleaner.get_statistics(cleaned_events)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   * {key}: {value:.1f}")
        else:
            print(f"   * {key}: {value}")
    
    # Afficher un aperçu
    print("\n[4] Sample Events Preview:")
    for i, event in enumerate(cleaned_events[:3], 1):
        print(f"\n   Event {i}:")
        print(f"   - Title: {event['title']}")
        print(f"   - Location: {event['location']}")
        print(f"   - Date: {event['date_start']}")
    
    print("\n" + "=" * 60)
    print("OK Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
