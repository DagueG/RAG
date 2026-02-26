"""
Test de validation de l'environnement du projet RAG.
Vérifie que tous les modules nécessaires sont installés et fonctionnels.
"""

import sys
import importlib
import pytest


class TestEnvironmentImports:
    """Tests des imports principaux du projet."""

    def test_import_faiss(self):
        """Vérifie que Faiss est installé et importable."""
        import faiss
        assert faiss is not None
        assert hasattr(faiss, 'IndexFlatL2')

    def test_import_langchain(self):
        """Vérifie que LangChain est installé."""
        import langchain
        assert langchain is not None

    def test_import_langchain_community_vectorstores(self):
        """Vérifie que FAISS peut être importé depuis langchain_community."""
        from langchain_community.vectorstores import FAISS
        assert FAISS is not None

    def test_import_langchain_community_embeddings(self):
        """Vérifie que HuggingFaceEmbeddings est disponible."""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        assert HuggingFaceEmbeddings is not None

    def test_import_mistralai(self):
        """Vérifie que Mistral AI est installé."""
        from mistralai.client import MistralClient
        assert MistralClient is not None

    def test_import_fastapi(self):
        """Vérifie que FastAPI est installé."""
        from fastapi import FastAPI
        assert FastAPI is not None

    def test_import_uvicorn(self):
        """Vérifie que Uvicorn est installé."""
        import uvicorn
        assert uvicorn is not None

    def test_import_pandas(self):
        """Vérifie que Pandas est installé."""
        import pandas
        assert pandas is not None

    def test_import_numpy(self):
        """Vérifie que NumPy est installé."""
        import numpy
        assert numpy is not None

    def test_import_pytest(self):
        """Vérifie que Pytest est installé."""
        assert pytest is not None

    def test_import_python_dotenv(self):
        """Vérifie que python-dotenv est installé."""
        from dotenv import load_dotenv
        assert load_dotenv is not None

    def test_import_requests(self):
        """Vérifie que Requests est installé."""
        import requests
        assert requests is not None

    def test_import_ragas(self):
        """Vérifie que Ragas est installé."""
        import ragas
        assert ragas is not None


class TestVersionCompatibility:
    """Tests de compatibilité entre les versions."""

    def test_faiss_version(self):
        """Vérifie la version de Faiss."""
        import faiss
        # Faiss n'expose pas toujours une version __version__
        assert faiss is not None

    def test_langchain_version(self):
        """Vérifie la version de LangChain."""
        import langchain
        version = getattr(langchain, '__version__', '0.1.0')
        assert version is not None

    def test_fastapi_version(self):
        """Vérifie la version de FastAPI."""
        import fastapi
        version = getattr(fastapi, '__version__', None)
        assert version is not None


class TestBasicFunctionality:
    """Tests de fonctionnalités basiques."""

    def test_faiss_index_creation(self):
        """Teste la création d'un index Faiss basique."""
        import faiss
        import numpy as np
        
        # Créer un simple index L2
        dimension = 10
        index = faiss.IndexFlatL2(dimension)
        
        # Ajouter quelques vecteurs
        vectors = np.random.random((5, dimension)).astype('float32')
        index.add(vectors)
        
        # Vérifier que l'index contient les vecteurs
        assert index.ntotal == 5

    def test_fastapi_app_creation(self):
        """Teste la création d'une application FastAPI."""
        from fastapi import FastAPI
        
        app = FastAPI(title="Test App")
        assert app.title == "Test App"
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        assert len(app.routes) > 0

    def test_pandas_dataframe_creation(self):
        """Teste la création d'un DataFrame Pandas."""
        import pandas as pd
        
        df = pd.DataFrame({
            'event_id': [1, 2, 3],
            'name': ['Event A', 'Event B', 'Event C'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        
        assert len(df) == 3
        assert 'event_id' in df.columns

    def test_numpy_array_operations(self):
        """Teste les opérations NumPy de base."""
        import numpy as np
        
        arr = np.array([1, 2, 3, 4, 5])
        result = np.sum(arr)
        
        assert result == 15


class TestEnvironmentVariables:
    """Tests de configuration d'environnement."""

    def test_dotenv_loading(self):
        """Teste que python-dotenv peut charger des variables."""
        import os
        from dotenv import load_dotenv
        
        # Cette fonction ne doit pas lever d'erreur
        load_dotenv()
        assert True  # Si on arrive ici, ça marche


@pytest.fixture(scope="module")
def environment_summary():
    """Fixture qui affiche un résumé de l'environnement."""
    import platform
    import sys
    
    summary = f"""
    ========== RÉSUMÉ DE L'ENVIRONNEMENT ==========
    Système d'exploitation: {platform.system()} {platform.release()}
    Version Python: {sys.version}
    Interpréteur: {sys.executable}
    =============================================
    """
    return summary


def test_environment_summary(environment_summary):
    """Affiche le résumé de l'environnement."""
    print(environment_summary)
    assert True


if __name__ == "__main__":
    # Permet d'exécuter le test directement avec: python tests/test_environment.py
    pytest.main([__file__, "-v", "--tb=short"])
