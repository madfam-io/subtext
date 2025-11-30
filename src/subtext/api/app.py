"""
Subtext FastAPI Application

Main application factory and configuration.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from subtext import __version__
from subtext.config import settings

from .routes import auth, billing, health, realtime, sessions, signals, webhooks

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# Lifespan Management
# ══════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info(
        "Starting Subtext API",
        version=__version__,
        environment=settings.app_env,
    )

    # Initialize database connection pool
    from subtext.db import init_db, close_db

    await init_db()

    # Initialize Redis
    from subtext.db.redis import init_redis, close_redis

    await init_redis()

    # Initialize ESP broadcaster
    from subtext.realtime.broadcaster import broadcaster as esp_broadcaster

    await esp_broadcaster.start()

    logger.info("Subtext API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Subtext API")

    # Stop ESP broadcaster
    await esp_broadcaster.stop()

    await close_db()
    await close_redis()
    logger.info("Subtext API shutdown complete")


# ══════════════════════════════════════════════════════════════
# Application Factory
# ══════════════════════════════════════════════════════════════


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Subtext API",
        description="Conversational Intelligence Infrastructure - Read the room, not just the transcript",
        version=__version__,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    # ──────────────────────────────────────────────────────────
    # Middleware
    # ──────────────────────────────────────────────────────────

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests with timing."""
        import time

        start = time.perf_counter()

        response = await call_next(request)

        duration = (time.perf_counter() - start) * 1000
        logger.info(
            "Request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration, 2),
        )

        return response

    # ──────────────────────────────────────────────────────────
    # Routes
    # ──────────────────────────────────────────────────────────

    # Health checks (no auth required)
    app.include_router(
        health.router,
        tags=["Health"],
    )

    # API v1 routes
    api_prefix = f"/api/{settings.api_version}"

    app.include_router(
        auth.router,
        prefix=f"{api_prefix}/auth",
        tags=["Authentication"],
    )

    app.include_router(
        sessions.router,
        prefix=f"{api_prefix}/sessions",
        tags=["Sessions"],
    )

    app.include_router(
        signals.router,
        prefix=f"{api_prefix}/signals",
        tags=["Signals"],
    )

    app.include_router(
        billing.router,
        prefix=f"{api_prefix}/billing",
        tags=["Billing"],
    )

    app.include_router(
        webhooks.router,
        prefix=f"{api_prefix}/webhooks",
        tags=["Webhooks"],
    )

    # WebSocket routes
    app.include_router(
        realtime.router,
        prefix="/ws",
        tags=["Realtime"],
    )

    return app


# Create default app instance
app = create_app()
