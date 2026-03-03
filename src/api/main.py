"""
FastAPI application for RAG system.
Exposes endpoints for question answering and index management.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from src.rag.rag_chain import RAGChain

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s:%(levelname)s:%(message)s'
)


# ================== REQUEST/RESPONSE MODELS ==================

class QuestionRequest(BaseModel):
    """Request model for /ask endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Je cherche un concert de jazz à Toulouse",
                "k": 5
            }
        }
    )
    question: str = Field(..., description="User question in French")
    k: Optional[int] = Field(5, description="Number of events to retrieve (default: 5)")


class RebuildRequest(BaseModel):
    """Request model for /rebuild endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "index_dir": "data/faiss_index"
            }
        }
    )
    index_dir: Optional[str] = Field("data/faiss_index", description="Path to index directory")


class EventData(BaseModel):
    """Event metadata returned in response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "1",
                "title": "Concert de Jazz",
                "date_start": "2026-03-15",
                "location": "Théâtre du Capitole, Toulouse",
                "description": "Concert de jazz contemporain"
            }
        }
    )
    id: str
    title: str
    date_start: str
    location: str
    description: Optional[str] = None


class AnswerResponse(BaseModel):
    """Response model for /ask endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "Voici les concerts de jazz à Toulouse...",
                "query": "Je cherche un concert de jazz",
                "events_retrieved": 3,
                "events": [
                    {
                        "id": "1",
                        "title": "Concert de Jazz",
                        "date_start": "2026-03-15",
                        "location": "Théâtre du Capitole, Toulouse"
                    }
                ],
                "model_used": "mistral-small"
            }
        }
    )
    response: str = Field(..., description="Generated answer from Mistral")
    query: str = Field(..., description="Original user question")
    events_retrieved: int = Field(..., description="Number of events used")
    events: list = Field(..., description="Retrieved events metadata")
    model_used: str = Field(..., description="LLM model name")


class RebuildResponse(BaseModel):
    """Response model for /rebuild endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Index rebuilt successfully",
                "events_indexed": 5
            }
        }
    )
    success: bool = Field(..., description="Whether rebuild was successful")
    message: str = Field(..., description="Status message")
    events_indexed: int = Field(..., description="Number of events indexed")


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "rag_initialized": True
            }
        }
    )
    status: str = Field(..., description="Health status")
    rag_initialized: bool = Field(..., description="Whether RAG is ready")


# ================== GLOBAL STATE ==================

# Global RAG Chain instance
rag_chain: Optional[RAGChain] = None


# ================== LIFESPAN ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage startup and shutdown events.
    """
    global rag_chain
    
    # Startup
    logger.info("Starting up FastAPI application...")
    
    # Load environment variables from .env file
    load_dotenv()
    logger.info("Environment variables loaded from .env")
    
    try:
        rag_chain = RAGChain(index_dir="data/faiss_index")
        logger.info("RAG Chain initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Chain: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    rag_chain = None


# ================== FASTAPI APP ==================

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation for cultural event recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== ENDPOINTS ==================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Status of the API and RAG system
    """
    return HealthResponse(
        status="healthy",
        rag_initialized=rag_chain is not None
    )


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """
    Ask a question and get an AI-generated response with event recommendations.
    
    Args:
        request: QuestionRequest containing the question and optional k parameter
    
    Returns:
        AnswerResponse: Generated response with retrieved events
    
    Raises:
        HTTPException: If question is empty or RAG is not initialized
    """
    global rag_chain
    
    if rag_chain is None:
        logger.error("RAG Chain not initialized")
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Try calling /rebuild"
        )
    
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Processing question: {request.question[:50]}...")
        
        # Generate response using RAG
        result = rag_chain.generate_response(
            query=request.question,
            k=request.k,
            include_context=True
        )
        
        logger.info(f"Response generated successfully")
        
        return AnswerResponse(
            response=result.get("response", ""),
            query=request.question,
            events_retrieved=result.get("num_events_retrieved", 0),
            events=result.get("events", []),
            model_used=result.get("model", "mistral-small")
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.post("/rebuild", response_model=RebuildResponse, tags=["Management"])
async def rebuild_index(request: RebuildRequest = None) -> RebuildResponse:
    """
    Rebuild the Faiss index from scratch.
    Useful after updating data or if index becomes corrupted.
    
    Args:
        request: RebuildRequest with optional index directory path
    
    Returns:
        RebuildResponse: Status of rebuild operation
    
    Raises:
        HTTPException: If rebuild fails
    """
    global rag_chain
    
    if rag_chain is None:
        logger.error("RAG Chain not initialized")
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        index_dir = request.index_dir if request else "data/faiss_index"
        logger.info(f"Rebuilding index from {index_dir}...")
        
        success = rag_chain.reload_index()
        
        if not success:
            raise Exception("Index reload returned False")
        
        num_events = len(rag_chain.events_metadata)
        logger.info(f"Index rebuilt successfully with {num_events} events")
        
        return RebuildResponse(
            success=True,
            message="Index rebuilt successfully",
            events_indexed=num_events
        )
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error rebuilding index: {str(e)}"
        )


@app.get("/info", tags=["Info"])
async def get_info() -> Dict[str, Any]:
    """
    Get information about the RAG system.
    
    Returns:
        Dictionary with system information
    """
    if rag_chain is None:
        return {
            "status": "not_initialized",
            "message": "RAG system not initialized"
        }
    
    return {
        "status": "ready",
        "events_indexed": len(rag_chain.events_metadata),
        "model": rag_chain.model_name,
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "vector_dimension": 384
    }


# ================== ROOT ROUTE ==================

@app.get("/", tags=["Info"])
async def root() -> Dict[str, str]:
    """
    Root endpoint with API information.
    """
    return {
        "message": "RAG System API",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
