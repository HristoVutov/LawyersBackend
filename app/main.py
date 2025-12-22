"""
FastAPI Application Entry Point.

This is the main entry point for the Python agent backend.
Run with: uvicorn app.main:app --reload
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.api import indexing, feedback
from app.agents import initialize_agent_registry
from app.services.conversation_logger import get_conversation_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    settings = get_settings()
    
    # Initialize conversation logger early
    logger = get_conversation_logger()
    
    # Startup
    print("🚀 Starting Lawyers Dashboard Agent Backend...")
    logger.log_terminal("🚀 Starting Lawyers Dashboard Agent Backend...")
    print(f"   📁 Project root: {settings.project_root}")
    print(f"   📂 Indexed files: {settings.indexed_files_dir}")

    # Initialize agents
    initialize_agent_registry()
    logger.log_terminal("✅ Agent registry initialized")
    
    
    if settings.langsmith_enabled:
        print("   📊 LangSmith tracing: ENABLED")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    else:
        print("   📊 LangSmith tracing: DISABLED (no API key)")
    
    if not settings.google_api_key:
        print("   ⚠️  WARNING: GOOGLE_API_KEY not set!")
    else:
        print("   🔑 Google API key: SET")
    
    yield
    
    # Shutdown
    print("👋 Shutting down agent backend...")


# Create FastAPI app
app = FastAPI(
    title="Lawyers Dashboard Agent API",
    description="Python backend for multi-agent legal assistant system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for Electron
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Electron file:// origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        logger = get_conversation_logger()
        
        # Log incoming request
        print(f"📥 {request.method} {request.url.path}")
        logger.log_terminal(f"📥 {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        # Log response with timing
        duration_ms = (time.time() - start_time) * 1000
        status_emoji = "✅" if response.status_code < 400 else "❌"
        log_msg = f"{status_emoji} {request.method} {request.url.path} → {response.status_code} ({duration_ms:.1f}ms)"
        print(log_msg)
        logger.log_terminal(log_msg)
        
        return response


app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(ws_router, prefix="/ws", tags=["websocket"])
app.include_router(indexing.router, prefix="/api", tags=["indexing"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Lawyers Dashboard Agent API",
        "version": "0.1.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
