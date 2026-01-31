"""
Lawyers Analytics - FastAPI Application Entry Point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import HealthResponse
from app.routes.events import router as events_router
from app.routes.users import router as users_router
from app.routes.usage import router as usage_router
from app.routes.payments import router as payments_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("🚀 Lawyers Analytics starting...")
    yield
    # Shutdown
    print("👋 Lawyers Analytics shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Lawyers Analytics",
        description="Cloud analytics service for Lawyers Dashboard usage tracking",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS
    origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Routes
    app.include_router(events_router)
    app.include_router(users_router)
    app.include_router(usage_router)
    app.include_router(payments_router)
    
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse()
    
    return app


# Application instance
app = create_app()
