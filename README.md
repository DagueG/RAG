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

## 🚀 Installation et démarrage

### Prérequis

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

#### 4. Installer les dépendances
```bash
uv pip install -r requirements.txt
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
# Réinstallez les dépendances
uv pip install -r requirements.txt --force-reinstall
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
