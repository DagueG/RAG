# System RAG pour Recommandation d'Événements Culturels

## 📋 Vue d'ensemble

Ce projet implémente un système **RAG (Retrieval-Augmented Generation)** pour Puls-Events. Le système permet à un chatbot intelligent de répondre à des questions sur les événements culturels à venir en combinant :

- **Récupération vectorielle** : recherche sémantique d'événements pertinents dans une base Faiss
- **Génération augmentée** : génération de réponses enrichies via le modèle Mistral
- **API REST** : exposition du système pour utilisation par les équipes métier

---

## 🎯 Objectifs du projet

✅ Développer un système RAG fonctionnel intégrant LangChain, Mistral et Faiss  
✅ Fournir une API REST pour interroger le système  
✅ Créer une base de données vectorielle d'événements culturels  
✅ Évaluer la qualité des réponses générées  
✅ Conteneuriser et déployer le système avec Docker  
✅ Présenter une démonstration live et documentée  

---

## 🏗️ Architecture du projet

```
RAG/
├── src/
│   ├── data_processing/      # Scripts de récupération et nettoyage
│   │   └── fetch_events.py   # Récupération API OpenAgenda
│   │   └── clean_data.py     # Nettoyage et préparation
│   │
│   ├── vectorization/        # Scripts de vectorisation
│   │   └── build_index.py    # Construction de l'index Faiss
│   │   └── embeddings.py     # Gestion des embeddings
│   │
│   ├── rag_system/           # Cœur du système RAG
│   │   └── rag_chain.py      # Orchestration LangChain
│   │   └── retriever.py      # Logique de récupération
│   │
│   └── api/                  # API REST
│       └── main.py           # Endpoint FastAPI
│
├── tests/                    # Tests unitaires (pytest)
│   ├── test_data_processing.py
│   ├── test_vectorization.py
│   ├── test_rag_system.py
│   └── test_api.py
│
├── data/                     # Données
│   ├── raw/                  # Données brutes
│   └── processed/            # Données traitées
│
├── docs/                     # Documentation
│   └── technical_report.md   # Rapport technique
│
├── requirements.txt          # Dépendances Python
├── .env.example             # Template de variables d'environnement
├── .gitignore               # Fichiers à ignorer
├── Dockerfile               # Conteneurisation
└── README.md               # Ce fichier
```

---

## 🚀 Instructions complètes du projet

### Prérequis

- **Python 3.11** 
- **Docker** (pour conteneurisation)
- **uv** package manager

### 1️⃣ Installation locale

```powershell
# Activer l'environnement virtuel
.\.venv\Scripts\Activate.ps1

# Synchroniser les dépendances
uv sync
```

### 2️⃣ Exécution complète du pipeline (dans l'ordre)

```powershell
# Étape 2: Récupérer les événements (100 événements Toulouse, futur, < 1 an)
python src/data_processing/fetch_events.py
# Output: data/raw/toulouse_events_raw.json (100 événements)

# Étape 3: Nettoyer et transformer les données
python src/data_processing/clean_data.py
# Output: data/processed/toulouse_events.json (100 événements nettoyés)

# Étape 4: Construire l'index Faiss
python src/vectorization/build_index.py
# Output: data/faiss_index/ (index + métadonnées)

# Étape 5: Lancer tous les tests
pytest tests/ -v

# Étape 5b: Lancer SEULEMENT les tests de validation des données
pytest tests/test_data_processing.py::TestFetchedEventsValidation -v

# Étape 5c: Lancer l'API REST (dans un nouveau terminal)
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# API disponible: http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

### 3️⃣ Déploiement Docker (Étape 6)

**Build l'image Docker:**
```powershell
docker build -t rag-api:latest .
```

**Lance le conteneur:**
```powershell
docker run -p 8000:8000 -it rag-api:latest
```

**Accès au conteneur:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

**Arrêter le conteneur:**
```powershell
# Ctrl+C dans le terminal, ou dans un autre terminal:
docker stop <container_id>
```

---

## 🏗️ Architecture complète

```
src/
├── data_processing/
│   ├── fetch_events.py       # Récupère 100 événements Toulouse (OpenDataSoft API)
│   └── clean_data.py         # Nettoie et transforme les données
│
├── vectorization/
│   ├── build_index.py        # Construit l'index Faiss (99 vecteurs)
│   └── embeddings.py         # Sentence-transformers (384-dim)
│
├── rag/
│   └── rag_chain.py          # Orchestration Faiss + Mistral LLM
│
└── api/
    └── main.py               # FastAPI avec 5 endpoints
```

---

## 📊 Résumé des étapes du projet

| Étape | Script | Input | Output | Status |
|-------|--------|-------|--------|--------|
| 2 | `fetch_events.py` | OpenDataSoft API | 100 événements bruts | ✅ |
| 3 | `clean_data.py` | Bruts | 100 événements nettoyés | ✅ |
| 4 | `build_index.py` | Nettoyés | Index Faiss 99 vecteurs | ✅ |
| 5 | `pytest tests/` | All tests | 72 tests PASSED | ✅ |
| 5 | `uvicorn src.api.main:app` | HttpRequests | Réponses RAG | ✅ |
| 6 | `docker build` | Dockerfile | Image Docker | ✅ |
| 6 | `docker run` | Image | Conteneur API | ✅ |

---

## 🧪 Tests

**Tous les tests:**
```powershell
pytest tests/ -v
# 72+ tests PASSED
```

**Validation des données:**
```powershell
pytest tests/test_data_processing.py::TestFetchedEventsValidation -v
# Vérifie: Toulouse, futures dates, < 1 an
```

**Tests RAG Chain:**
```powershell
pytest tests/test_rag_chain.py -v
```

---

## 🌐 API REST - Utilisation et Test

### Lancer l'API

```powershell
# Terminal 1: Lancer le serveur API
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# API disponible à: http://localhost:8000
# Documentation interactive: http://localhost:8000/docs
# OpenAPI JSON: http://localhost:8000/openapi.json
```

### Endpoints disponibles

#### 1️⃣ **GET /health**
Vérifie l'état de l'API et du système RAG.

```bash
curl http://localhost:8000/health
```

#### 2️⃣ **POST /ask**
Poser une question et obtenir une réponse avec événements recommandés.

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Je cherche un concert de jazz à Toulouse", "k": 5}'
```

#### 3️⃣ **POST /rebuild**
Reconstruire l'index Faiss (utile après mise à jour des données).

#### 4️⃣ **GET /info**
Observer les informations du système.

### Test interactif de l'API

```powershell
# Terminal 2: Lancer le script de démonstration
python tests/demo_api.py
```

Ce script teste automatiquement les endpoints et affiche les réponses.

---

## 🎯 Évaluation Automatique avec Ragas

Evaluez la qualité des réponses générées par le système RAG.

### Métriques d'évaluation

| Métrique | Définition |
|----------|----------|
| **Faithfulness** | Fidélité aux données sources récupérées |
| **Answer Relevance** | Pertinence de la réponse par rapport à la question |
| **Context Precision** | Pertinence des événements sélectionnés |
| **Context Recall** | Couverture des événements pertinents |

### Lancer l'évaluation

```powershell
# Terminal 1: L'API doit être en cours d'exécution
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Lancer le script d'évaluation
python tests/evaluate_rag.py
```

Le script génère :
- Scores détaillés pour chaque question
- Statistiques agrégées
- Résultats sauvegardés dans `tests/evaluation_results.json`
- Recommandations d'amélioration

### Dataset d'évaluation

Le dataset se trouve dans [`tests/evaluation_dataset.json`](tests/evaluation_dataset.json) avec :
- 5 questions de test annotées
- Réponses de référence
- Mots-clés attendus

---

## 📝 Fichiers importants

- [`fetch_events.py`](src/data_processing/fetch_events.py#L45-L95) - Filtre à l'API (Toulouse + futur < 1 an)
- [`clean_data.py`](src/data_processing/clean_data.py) - Support formats OpenDataSoft + OpenAgenda
- [`rag_chain.py`](src/rag/rag_chain.py) - Mistral + Faiss orchestration
- [`main.py`](src/api/main.py) - FastAPI endpoints avec gestion d'erreurs
- [`demo_api.py`](tests/demo_api.py) - Test interactif de l'API
- [`evaluate_rag.py`](tests/evaluate_rag.py) - Évaluation automatique avec Ragas
- [`evaluation_dataset.json`](tests/evaluation_dataset.json) - Dataset d'évaluation annoté
- [`Dockerfile`](Dockerfile) - Multi-stage build
- [`pyproject.toml`](pyproject.toml) - Dépendances uv

---

## Prérequis

- **Python >= 3.8** installé sur votre machine
- **uv** (gestionnaire de packages) - [Installation](https://github.com/astral-sh/uv)
- Connexion Internet stable

### Étapes d'installation

#### 1. Cloner ou télécharger le projet
```bash
cd z:\openclassroom\RAG
```

#### 2. Créer un environnement virtuel
```bash
uv venv
```

#### 3. Activer l'environnement virtuel

**Sur PowerShell :**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Sur CMD :**
```cmd
.venv\Scripts\activate.bat
```

**Sur Linux/Mac :**
```bash
source venv/bin/activate
```

#### 4. Synchroniser les dépendances
```bash
uv sync
```

#### 5. Configurer les variables d'environnement
```bash
# Copier le template
cp .env.example .env

# Éditer .env et ajouter vos clés API
# MISTRAL_API_KEY=votre_clef_ici
```

#### 6. Tester l'installation
```bash
python -c "import faiss; from langchain.vectorstores import FAISS; from langchain.embeddings import HuggingFaceEmbeddings; from mistralai import MistralClient; print('✅ Tous les modules importent correctement !')"
```

---

## 📚 Utilisation

### Construire la base de données vectorielle
```bash
python src/vectorization/build_index.py
```

### Lancer l'API REST
```bash
uvicorn src.api.main:app --reload
```

L'API sera accessible à `http://localhost:8000`

Documentation interactive : `http://localhost:8000/docs`

### Tester l'API
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les événements jazz à Paris cette semaine ?"}'
```

### Lancer les tests
```bash
pytest tests/ -v --cov=src
```

---

## ⚙️ Technologies utilisées

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Récupération | OpenAgenda API | Données d'événements |
| Manipulation | Pandas, NumPy | Nettoyage et préparation |
| Vecteurs | HuggingFaceEmbeddings | Conversion texte → vecteurs |
| Base vectorielle | Faiss | Stockage et recherche efficace |
| Orchestration | LangChain | Chaînes de traitement RAG |
| LLM | Mistral | Génération de réponses |
| API | FastAPI + Uvicorn | Exposition du système |
| Évaluation | Ragas | Métriques de qualité |
| Tests | Pytest | Validation automatisée |
| Conteneurisation | Docker | Déploiement reproductible |

---

## 📊 Pipeline RAG

```
Question utilisateur
        ↓
[Vecteur de la question]
        ↓
[Recherche Faiss - Top K événements similaires]
        ↓
[Contexte enrichi + Question]
        ↓
[Mistral - Génération de réponse]
        ↓
Réponse augmentée
```

---

## 🧪 Points de validation

Avant de passer à l'étape suivante, assurez-vous que :

- [ ] L'environnement virtuel est activé
- [ ] Toutes les dépendances s'installent sans erreur
- [ ] Les imports tests passent correctement
- [ ] Vous êtes capable d'activer/désactiver l'environnement

---

## 📖 Documentation additionnelle

- **Rapport technique** : voir `docs/technical_report.md` (à créer lors de l'étape finale)
- **Tests** : voir les fichiers dans `tests/`
- **API** : documentation interactive via `/docs` quand l'API tourne

---

## 🛠️ Dépannage

### ❌ "Module not found" lors des imports
```bash
# Vérifiez que l'environnement virtuel est activé
# Resynchronisez les dépendances
uv sync --reinstall
```

### ❌ "Mistral API Key not found"
```bash
# Vérifiez que le fichier .env existe et a une clé valide
cat .env
```

### ❌ Erreur Faiss lors de l'import
```bash
# Utilisez faiss-cpu au lieu de faiss-gpu
uv pip install faiss-cpu --force-reinstall
```

### ❌ "Cannot connect to API" lors de demo_api.py ou evaluate_rag.py
```bash
# Assurez-vous que l'API tourne dans un autre terminal
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

---

## 📝 Notes importantes

- ✅ Ne jamais commiter la clé API (utilisez `.env` ignoré par Git)
- ✅ Toujours utiliser un environnement virtuel isolé
- ✅ Tester l'installation sur une "nouvelle machine" (après suppression de cache)
- ✅ Le dossier `env` ou `venv` ne doit PAS être ajouté à Git

---

## 👥 Équipe

**Développeur** : Data Scientist freelance  
**Client** : Puls-Events  
**Responsable Technique** : Jérémy

---

## 📅 Historique des versions

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 26/02/2026 | Configuration initiale de l'environnement |

---

**Prêt à démarrer ? Consultez les étapes du projet dans `contexte.txt`**
