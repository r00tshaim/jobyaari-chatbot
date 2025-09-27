"""
FastAPI Server for JobYaari RAG Chatbot
=======================================

This module provides REST API endpoints for the JobYaari RAG chatbot.
It serves as the backend service that the Streamlit frontend will call.

Endpoints:
- POST /chat - Main chat endpoint
- GET /health - Health check endpoint  
- GET /info - System information endpoint
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our chatbot
from chatbot import get_chatbot, ChatbotConfig

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: Optional[str] = Field(None, description="Optional session ID")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict] = Field(default=[], description="Source documents")
    metadata: Dict = Field(default={}, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if any")

class HealthResponse(BaseModel):
    """Response model for health endpoint"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")

class InfoResponse(BaseModel):
    """Response model for info endpoint"""
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="Service description")
    endpoints: List[str] = Field(..., description="Available endpoints")
    config: Dict = Field(..., description="Current configuration")

# Global variables
chatbot_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app"""
    global chatbot_instance
    
    # Startup
    logger.info("Starting JobYaari RAG Chatbot Server")
    try:
        chatbot_instance = get_chatbot()
        logger.info("Chatbot initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down JobYaari RAG Chatbot Server")

# Create FastAPI app
app = FastAPI(
    title="JobYaari RAG Chatbot API",
    description="REST API for JobYaari job search chatbot using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with welcome message"""
    return {
        "message": "Welcome to JobYaari RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test chatbot availability
        if chatbot_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chatbot not initialized"
            )
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.get("/info", response_model=InfoResponse)
async def service_info():
    """Get service information"""
    try:
        config_info = {
            "llm_provider": os.getenv("AGENT_LLM_PROVIDER", "gemini"),
            "embedding_provider": os.getenv("AGENT_EMBEDDING_PROVIDER", "ollama"),
            "qdrant_url": os.getenv("AGENT_QDRANT_URL", "http://localhost:6333"),
            "qdrant_collection": os.getenv("AGENT_QDRANT_COLLECTION", "jobyaari_jobs"),
            "max_results": os.getenv("AGENT_MAX_RESULTS", "5"),
        }
        
        return InfoResponse(
            service_name="JobYaari RAG Chatbot",
            version="1.0.0",
            description="Retrieval-Augmented Generation chatbot for JobYaari job postings",
            endpoints=["/", "/health", "/info", "/chat"],
            config=config_info
        )
        
    except Exception as e:
        logger.error(f"Error getting service info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving service info: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for processing user queries"""
    logger.info(f"Received chat request: {request.query}")
    
    try:
        # Validate chatbot availability
        if chatbot_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chatbot service not available"
            )
        
        # Process the query
        response = chatbot_instance.chat(request.query)
        
        # Add request metadata
        response["metadata"]["session_id"] = request.session_id
        response["metadata"]["request_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Chat request processed successfully. Answer length: {len(response['answer'])}")
        
        return ChatResponse(**response)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        # Return error response instead of raising exception
        error_response = {
            "answer": f"I encountered an error while processing your request: {str(e)}",
            "sources": [],
            "metadata": {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "session_id": request.session_id
            },
            "error": str(e)
        }
        return ChatResponse(**error_response)

# Additional utility endpoints

@app.get("/config")
async def get_config():
    """Get current configuration (non-sensitive info only)"""
    try:
        return {
            "llm_provider": os.getenv("AGENT_LLM_PROVIDER", "gemini"),
            "embedding_provider": os.getenv("AGENT_EMBEDDING_PROVIDER", "ollama"),
            "embedding_model": os.getenv("AGENT_EMBEDDING_MODEL", "nomic-embed-text:latest"),
            "qdrant_collection": os.getenv("AGENT_QDRANT_COLLECTION", "jobyaari_jobs"),
            "max_results": os.getenv("AGENT_MAX_RESULTS", "5"),
            "similarity_threshold": os.getenv("AGENT_SIMILARITY_THRESHOLD", "0.7")
        }
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving configuration: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get basic statistics"""
    try:
        # This is a placeholder - you could add actual statistics
        return {
            "service_uptime": "N/A",
            "total_requests": "N/A", 
            "successful_requests": "N/A",
            "failed_requests": "N/A",
            "average_response_time": "N/A"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving statistics: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": f"The requested endpoint was not found",
            "available_endpoints": ["/", "/health", "/info", "/chat", "/config", "/stats"]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An internal error occurred while processing your request"
        }
    )

# Development/testing utilities
if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("AGENT_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_SERVER_PORT", "8000"))
    reload = os.getenv("AGENT_SERVER_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    try:
        uvicorn.run(
            "server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise