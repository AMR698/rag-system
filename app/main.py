"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config.settings import configure_logging, get_settings

configure_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="RAG backend module for an educational game: retrieval, adaptive question "
    "generation, answer evaluation, and performance reporting.",
    version="1.0.0",
)

# CORS is left permissive here because the frontend/game-client origin is
# not part of this module's scope. Restrict `allow_origins` in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler so unexpected errors never leak a raw traceback to clients."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please check server logs."},
    )


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("%s starting up...", settings.APP_NAME)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("%s shutting down...", settings.APP_NAME)
