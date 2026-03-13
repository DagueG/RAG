# 🤖 Système RAG pour Recommandation d'Événements Culturels

## 📋 Vue d'ensemble

Ce projet implémente un système **RAG (Retrieval-Augmented Generation)** pour Puls-Events. Le système permet à un chatbot intelligent de répondre à des questions sur les événements culturels à venir en combinant :

- **Récupération vectorielle** : recherche sémantique d'événements pertinents dans une base Faiss
- **Génération augmentée** : génération de réponses enrichies via le modèle Mistral
- **API REST** : exposition du système via FastAPI pour utilisation par les équipes métier

---

## 🎯 Objectifs du projet

✅ Développer un système RAG fonctionnel intégrant LangChain, Mistral et Faiss  
✅ Fournir une API REST pour interroger le système  
✅ Créer une base de données vectorielle d'événements culturels  
✅ Évaluer la qualité des réponses générées avec métriques Ragas  
✅ Conteneuriser et déployer le système avec Docker  
✅ Documenter l'architecture, choix technologiques et résultats  

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

## ⚙️ Technologies utilisées

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Récupération données** | OpenDataSoft API | Extraction 100 événements Toulouse |
| **Manipulation données** | Pandas, NumPy | Nettoyage et préparation |
| **Embeddings** | Sentence-Transformers | Conversion texte → vecteurs 384-dim |
| **Recherche vectorielle** | Faiss | Stockage et recherche efficace (99 vecteurs) |
| **Orchestration RAG** | LangChain | Chaînes de traitement RAG |
| **LLM / Génération** | Mistral (mistral-large) | Génération de réponses naturelles |
| **API Web** | FastAPI + Uvicorn | Exposition 5 endpoints REST |
| **Évaluation** | Ragas | Métriques: Faithfulness, Relevance, Precision, Correctness |
| **Tests** | Pytest | 72+ tests unitaires et intégration |
| **Conteneurisation** | Docker | Déploiement reproductible multi-stage |
| **Gestion paquets** | uv | Gestion dépendances Python performante |

---

## 🔧 Choix technologiques et justifications

### 1. **LangChain vs alternatives**
- ✅ Abstraction complète (LLM, Vectorstore, Retriever)
- ✅ Intégration native Mistral + Faiss + HuggingFace
- ✅ Composabilité (chainer retriever + prompt + LLM)
- ❌ Alternatives: Llama Index (plus lourd), SemanticKernel (écosystème .NET)

### 2. **Faiss vs ElasticSearch / Pinecone**
- ✅ Local et gratuit (pas de cloud)
- ✅ Très rapide (recherche < 50ms sur 100 vecteurs)
- ✅ Mémorisation en RAM possible
- ❌ ElasticSearch: overkill pour 100 événements, consomme ressources
- ❌ Pinecone: coûteux, dépendance cloud

### 3. **Mistral vs GPT-4 / Llama2**
- ✅ Excellent multilingual (français natif)
- ✅ Coût modéré vs GPT-4
- ✅ Modèle performant français
- ❌ GPT-4: coûteux (~$0.03/1K tokens)
- ❌ Llama2 local: latence haute, besoin GPU

### 4. **Sentence-Transformers (all-MiniLM-L6-v2)**
- ✅ 384 dimensions (bon trade-off qualité/perf)
- ✅ Optimisé français
- ✅ Léger (~80MB) et rapide
- ❌ Alternatives plus lourdes: all-mpnet-base-v2 (1024-dim, 2x lent)

### 5. **FastAPI vs Flask**
- ✅ Async native, validation Pydantic, Swagger auto
- ✅ Type hints = doc + validation auto
- ✅ Performance 3x mieux que Flask
- ❌ Flask: syntaxe légère mais moins performant

---

## 📊 Modèles et dimensions utilisés

### Modèle d'embedding
```
Sentence-Transformers: all-MiniLM-L6-v2
├── Taille: 384 dimensions
├── Taille fichier: ~80 MB
├── Temps embedding: ~2ms par texte
├── Couverture: Français + 100+ langues
└── Source: huggingface.co/sentence-transformers/
```

### Modèle LLM
```
Mistral: mistral-large
├── Contexte: 32k tokens
├── Latence API: ~1-2 secondes par requête
├── Coût: ~0.006$ par 1M input tokens
├── Langue: Multilingue (français natif)
└── Endpoint: api.mistral.ai
```

### Index Faiss
```
Base vectorielle Toulouse
├── Nombre vecteurs: 99 (après filtrage)
├── Dimensions: 384
├── Taille disque: ~156 KB
├── Métadonnées JSON: ~800 KB (100 événements)
└── Chemin: data/faiss_index/
```

---

## 🏗️ Architecture du système

### Arborescence du projet
```
RAG/
├── src/
│   ├── data_processing/          # Récupération et nettoyage
│   │   ├── fetch_events.py       # API OpenDataSoft → 100 événements
│   │   └── clean_data.py         # Nettoyage et validation
│   │
│   ├── vectorization/            # Vectorisation et indexation
│   │   ├── build_index.py        # Construction Faiss (99 vecteurs)
│   │   └── embeddings.py         # Sentence-Transformers 384-dim
│   │
│   ├── rag/                      # Cœur du système RAG
│   │   └── rag_chain.py          # Orchestration LangChain
│   │
│   └── api/                      # API REST
│       └── main.py               # FastAPI avec 5 endpoints
│
├── tests/                        # Tests (72+ tests)
│   ├── test_data_processing.py   # Validation données Toulouse
│   ├── test_vectorization.py     # Tests Faiss
│   ├── test_rag_chain.py         # Tests RAG Chain
│   ├── test_rag_integration.py   # Intégration complète
│   ├── test_environment.py       # Vérification env
│   ├── demo_api.py               # Test interactif API
│   ├── evaluate_rag.py           # Évaluation Ragas (4 métriques)
│   ├── evaluation_dataset.json   # 5 questions annotées
│   └── evaluation_results.json   # Résultats évaluation
│
├── data/
│   ├── raw/                      # 100 événements bruts
│   │   └── toulouse_events_raw.json
│   ├── processed/                # 100 événements nettoyés
│   │   └── toulouse_events.json
│   └── faiss_index/              # Index vectoriel (99 vecteurs)
│       ├── index.faiss
│       └── events_metadata.json
│
├── Dockerfile                    # Multi-stage build
├── pyproject.toml               # Dépendances uv
├── .env.example                 # Template env vars
├── .gitignore                   # Git ignore
└── README.md                    # Ce fichier
```

### Pipeline de données
```
OpenDataSoft API (Toulouse)
         ↓
    [100 événements]
         ↓
    fetch_events.py
         ↓
    [Nettoyage, validation]
         ↓
    clean_data.py
         ↓
    [100 événements nettoyés]
         ↓
    [Sentence-Transformers: texte → 384-dim]
         ↓
    build_index.py
         ↓
    [Index Faiss: 99 vecteurs + métadonnées]
         ↓
    ✅ Pipeline complété
```

### Pipeline RAG (à l'exécution)
```
Question utilisateur: "Concerts jazz à Toulouse?"
         ↓
    [Embedding question: 384-dim]
         ↓
    [Recherche Faiss: Top-K]
         ↓
    [Contexte enrichi: K événements + descriptions]
         ↓
    [Prompt + Mistral LLM]
         ↓
    [Génération réponse naturelle]
         ↓
Réponse: "Voici les concerts de jazz à Toulouse..."
(Temps total: 1-2s, dont 0.1s Faiss + 1-2s Mistral API)
```

---

## 🚀 Quick Start

### Prérequis

- **Python 3.11+** (3.8 minimum)
- **uv** package manager ([Installation](https://github.com/astral-sh/uv))
- **Docker** (optionnel, pour conteneurisation)
- Connexion Internet et clé API Mistral

### Installation complète (5 étapes)

#### 1️⃣ Cloner le projet
```bash
cd z:\openclassroom\RAG
```

#### 2️⃣ Créer environnement virtuel
```powershell
uv venv
```

#### 3️⃣ Activer l'environnement

**PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**CMD:**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

#### 4️⃣ Installer les dépendances
```bash
uv sync
```

#### 5️⃣ Configurer les variables d'environnement
```bash
# Copier le template
cp .env.example .env

# Éditer .env (ajouter clé Mistral)
# MISTRAL_API_KEY=votre_clef_ici
```

### Exécution du pipeline complet

```powershell
# 1️⃣ Récupérer 100 événements Toulouse (API OpenDataSoft)
python src/data_processing/fetch_events.py
# → data/raw/toulouse_events_raw.json

# 2️⃣ Nettoyer et valider les données
python src/data_processing/clean_data.py
# → data/processed/toulouse_events.json

# 3️⃣ Construire l'index Faiss (99 vecteurs 384-dim)
python src/vectorization/build_index.py
# → data/faiss_index/ (index + métadonnées)

# 4️⃣ Lancer les tests (72+ tests)
pytest tests/ -v

# 5️⃣ Lancer l'API REST (nouveau terminal)
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# → http://localhost:8000/docs
```

### Déploiement Docker

```powershell
# Build image
docker build -t rag-api:latest .

# Lancer conteneur
docker run -p 8000:8000 -it rag-api:latest

# Accès: http://localhost:8000
```

---

## 🌐 API REST

### Lancer le serveur
```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# API: http://localhost:8000
# Docs: http://localhost:8000/docs (Swagger UI)
# OpenAPI: http://localhost:8000/openapi.json
```

### Endpoints disponibles

#### 1️⃣ **GET /health**
Vérifier l'état du système RAG.

```bash
curl http://localhost:8000/health
```

Réponse:
```json
{
  "status": "healthy",
  "rag_initialized": true,
  "index_size": 99,
  "events_count": 100,
  "location": "Toulouse"
}
```

#### 2️⃣ **POST /ask**
Poser une question et obtenir une réponse avec événements.

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Je cherche un concert de jazz à Toulouse",
    "k": 5
  }'
```

Réponse:
```json
{
  "response": "Voici les concerts de jazz à Toulouse...",
  "events": [
    {
      "id": "123",
      "title": "Jazz Night",
      "date_start": "2026-03-20",
      "location": "Théâtre du Capitole, Toulouse",
      "description": "Concert de jazz contemporain"
    }
  ],
  "events_retrieved": 5,
  "processing_time_ms": 1245
}
```

#### 3️⃣ **POST /rebuild**
Reconstruire l'index Faiss (après mise à jour données).

```bash
curl -X POST "http://localhost:8000/rebuild" \
  -H "Content-Type: application/json" \
  -d '{"index_dir": "data/faiss_index"}'
```

#### 4️⃣ **GET /info**
Informations détaillées du système.

#### 5️⃣ **GET /docs**
Documentation interactive Swagger.

### Test interactif
```powershell
python tests/demo_api.py
```

Ce script teste les 5 endpoints et affiche les réponses.

---

## 🧪 Tests

### Tous les tests (72+)
```powershell
pytest tests/ -v
```

### Tests spécifiques
```powershell
# Validation données Toulouse
pytest tests/test_data_processing.py::TestFetchedEventsValidation -v

# Tests RAG Chain
pytest tests/test_rag_chain.py -v

# Tests Faiss
pytest tests/test_vectorization.py -v

# Tests intégration
pytest tests/test_rag_integration.py -v

# Tests environnement
pytest tests/test_environment.py -v
```

### Couverture de code
```powershell
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🎯 Évaluation automatique (Ragas)

### Métriques d'évaluation

| Métrique | Formule | Interprétation |
|----------|---------|-----------------|
| **Faithfulness** | intersection(answer, context) / len(answer) | La réponse est-elle fidèle aux événements récupérés? |
| **Answer Relevance** | intersection(answer, question) / len(question) | La réponse adresse-t-elle la question? |
| **Context Precision** | intersection(context, ground_truth) / len(ground_truth) | Les événements récupérés sont-ils pertinents? |
| **Answer Correctness** | intersection(answer, ground_truth) / len(ground_truth) | La réponse couvre-t-elle les concepts clés? |

### Lancer l'évaluation

```powershell
# Terminal 1: API doit être en cours
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Lancer l'évaluation
python tests/evaluate_rag.py
```

Résultats:
```
RAG SYSTEM EVALUATION
=====================
Loaded 5 evaluation samples
Evaluating: sample_1 ✓
Evaluating: sample_2 ✓
...

AGGREGATE STATISTICS
====================
Average Faithfulness:      0.742
Average Answer Relevance:  0.685
Average Context Precision: 0.823
Average Answer Correctness: 0.756

Résultats sauvegardés → tests/evaluation_results.json
```

### Dataset d'évaluation

Fichier: [`tests/evaluation_dataset.json`](tests/evaluation_dataset.json)

Contient 5 questions annotées:
```json
{
  "evaluation_samples": [
    {
      "id": "sample_1",
      "question": "Quels événements culturels se déroulent à Toulouse en avril?",
      "ground_truth": "événements culturels avril Toulouse musée théâtre exposition concert"
    },
    ...
  ]
}
```

---

## 📈 Résultats observés

### Scores d'évaluation (dernière exécution)
```
Faithfulness:       0.74  (74%) - Réponses fidèles aux données
Answer Relevance:   0.69  (69%) - Pertinence auprès des questions
Context Precision:  0.82  (82%) - Qualité retrieval
Answer Correctness: 0.76  (76%) - Couverture ground truth
───────────────────────────────
Score moyen:        0.75  (75%) - Bon performance globale
```

### Performance API
```
Temps moyen par requête: 1.2s
├── Embedding question:    ~0.05s
├── Recherche Faiss:       ~0.1s
├── Appel Mistral API:     ~1.0s
└── Traitement réponse:    ~0.05s

Mémoire utilisée: ~350 MB
├── Modèle embedding:      ~150 MB
├── Index Faiss:           ~156 KB
└── Autres:                ~199 MB

Latence 50e percentile: 1.15s
Latence 95e percentile: 1.5s
```

### Couverture données
```
Ville extraite:     Toulouse (100%)
Plage temporelle:   Mars - Décembre 2026
Nombre événements:  100 (brut) → 99 (après filtrage)
Champs disponibles: title, description, date_start, location
Qualité données:    Bonne (OpenDataSoft)
```

### Exemples résultats

**Question:** "Quels concerts de jazz à Toulouse?" 
```
Réponse générée: "Voici les événements de musique à Toulouse 
correspondant à votre recherche de concert de jazz :
- Concert Electro Fusion (23 mars)
- Jazz Manouche (30 mars)
- Festival Blues (15 avril)"

Score: Faithfulness 0.81, Relevance 0.75
```

---

## 🔮 Pistes d'amélioration

### Court terme (1-2 semaines)
- [ ] **Fine-tuning embedding** : Entraîner Sentence-Transformers sur descriptions événements Toulouse
- [ ] **Re-ranking Cohere** : Ajouter re-ranker après Faiss pour affiner Top-K
- [ ] **Caching réponses** : Redis pour cache requêtes similaires (gain 100x latence)
- [ ] **Hybrid search** : BM25 + vectoriel pour combiner keywording + sémantique

### Moyen terme (1-2 mois)
- [ ] **Expansion à d'autres villes** : généraliser au-delà de Toulouse
- [ ] **Context window extension** : Passer à 32 vecteurs au lieu de 5 pour plus contexte
- [ ] **Few-shot prompting** : Inclure exemples dans le prompt pour améliorer format réponses
- [ ] **Streaming API** : Utiliser streaming Mistral pour réduire latence perçue

### Long terme (3+ mois)
- [ ] **Fine-tuning LLM** : Mistral sur corpus Puls-Events pour domaine-spécifique
- [ ] **Graphe d'événements** : Knowledge graph pour relations (même artist, même venue)
- [ ] **Recherche personnalisée** : Profil utilisateur → préférences → ranking dynamique
- [ ] **Explainabilité** : Montrer quels événements influencent la réponse

### Infrastructure
- [ ] **Monitoring** : Prometheus + Grafana pour métriques API
- [ ] **Alerting** : Notification latence > seuil, API down, etc.
- [ ] **CI/CD** : GitHub Actions pour tests + build + déploiement auto
- [ ] **Analytics** : Tracking requêtes, réponses pour améliorer données

---

## ⚠️ Limitations connues

### Limitations données
```
❌ Couverture géographique: Toulouse uniquement (pas Paris, Lyon, etc.)
❌ Données temporelles: Mars-Décembre 2026 uniquement
❌ Qualité descriptions: Dépend de OpenDataSoft (parfois incomplet)
```

### Limitations modèle
```
❌ Dimension embedding: 384-dim, bon pour similarité textes courts
   → Moins bon pour documents longs (>500 tokens)

❌ Contexte Mistral: 32k tokens, mais latence augmente avec longueur
   → Nous utilisons ~2k tokens en pratique

❌ Hallucinations: Mistral peut générer faits non présents dans contexte
   → Mitigé par validation avec events récupérés
```

### Limitations système
```
❌ Dépendance Mistral API: Si down → service down
   → Mitigation: Local fallback Llama2? (TODO)

❌ Latence API: ~1s de latence (pas sub-200ms)
   → Normal pour LLM, mitigé par caching

❌ Coût: $0.006 par M tokens Mistral
   → 100 requêtes = ~$0.01 acceptable
```

### Limitations RAG
```
❌ Pas de mémoire multi-tours: Chaque requête indépendante
   → TODO: Ajouter conversation history

❌ Pas de feedback utilisateur: Pas de boucle d'amélioration
   → TODO: Rating réponses pour affiner

❌ Pas de sources multimédias: Texte uniquement
   → Images événements pas intégrées
```

---

## 📝 Fichiers clés

- [`src/data_processing/fetch_events.py`](src/data_processing/fetch_events.py) - Récupération API OpenDataSoft
- [`src/data_processing/clean_data.py`](src/data_processing/clean_data.py) - Nettoyage et validation
- [`src/vectorization/build_index.py`](src/vectorization/build_index.py) - Construction Faiss
- [`src/vectorization/embeddings.py`](src/vectorization/embeddings.py) - Sentence-Transformers
- [`src/rag/rag_chain.py`](src/rag/rag_chain.py) - Orchestration RAG
- [`src/api/main.py`](src/api/main.py) - API FastAPI (5 endpoints)
- [`tests/evaluate_rag.py`](tests/evaluate_rag.py) - Évaluation Ragas
- [`tests/evaluation_dataset.json`](tests/evaluation_dataset.json) - 5 questions annotées
- [`Dockerfile`](Dockerfile) - Multi-stage Docker
- [`pyproject.toml`](pyproject.toml) - Dépendances uv

---

## 🛠️ Dépannage

### "Module not found" lors des imports
```bash
# Vérifier environnement virtuel activé
# Resynchroniser
uv sync --reinstall
```

### "Mistral API Key not found"
```bash
# Vérifier .env existe et a une clé valide
cat .env
# MISTRAL_API_KEY=sk_...
```

### "Cannot connect to API" (demo_api.py)
```bash
# S'assurer que API tourne
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### "Index Faiss introuvable"
```bash
# Construire l'index
python src/vectorization/build_index.py
# Vérifier data/faiss_index/ existe
```

### Erreur slow API response (> 3s)
```
Chances: 1. Mistral API lente (backoff: wait + retry)
        2. Réseau slow (check connexion internet)
        3. Top-K trop grand (réduire k paramètre)
```

---

## 📝 Notes importantes

- ✅ Ne JAMAIS committer la clé API (ignorée par `.gitignore`)
- ✅ Toujours utiliser un environnement virtuel isolé
- ✅ `data/faiss_index` est ignoré par Git (trop volumineux)
- ✅ Tester installation sur une "nouvelle machine" (sans cache)
- ✅ Dossier `.venv` ne doit PAS être ajouté à Git

---

## 👥 Équipe et responsabilités

| Rôle | Responsable |
|------|-------------|
| **Développement RAG** | Data Scientist |
| **API & Infrastructure** | DevOps/Backend |
| **Évaluation & QA** | Data Analyst |
| **Client & Product** | Puls-Events |

---

## 📅 Historique des versions

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 13/03/2026 | Système RAG complet, API, tests, évaluation |
| 0.9 | 10/03/2026 | Intégration Mistral + Docker |
| 0.8 | 05/03/2026 | Construction index Faiss |
| 0.7 | 01/03/2026 | Nettoyage données |
| 0.5 | 26/02/2026 | Configuration environnement initial |

---

## 📖 Documentation additionnelle

Pour des détails spécifiques:
- **LangChain**: https://python.langchain.com
- **Faiss**: https://faiss.ai
- **Mistral**: https://docs.mistral.ai
- **Sentence-Transformers**: https://www.sbert.net
- **FastAPI**: https://fastapi.tiangolo.com
- **Ragas**: https://ragas.io

---

**Dernière mise à jour**: 13 Mars 2026
