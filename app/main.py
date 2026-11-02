"""FastAPI application entry point.

Run locally:  uvicorn app.main:app --reload --port 7860
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.deps import get_analyzer
from app.api.routes import router
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the models in the background so the server starts accepting requests immediately.
    threading.Thread(target=get_analyzer, daemon=True).start()
    yield


app = FastAPI(
    title=f"{settings.app_name} API",
    version=settings.version,
    description="AI-powered deepfake detection for images and videos, with Grad-CAM explanations.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
