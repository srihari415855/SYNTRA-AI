"""FastAPI Main Application Entry Point with Static UI Serving."""

from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.storage.database import init_db

# Initialize JSON logging
configure_logging()
logger = get_logger("ai_system.main")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown hooks."""
    logger.info("Initializing SYNTRA / OpenPilot AI Platform...", extra={"event_type": "STARTUP"})
    init_db()
    logger.info("Database schemas initialized.", extra={"event_type": "DB_INIT"})
    yield
    logger.info("Shutting down AI System...", extra={"event_type": "SHUTDOWN"})


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="SYNTRA AI — OpenPilot Engine: Real AI Orchestration and Model-Routing Platform",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(api_router, prefix="/api/v1")
app.include_router(api_router, prefix="")  # Exposes /health


# Mount Static Files and Root Web App
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serves the single-page application frontend."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "SYNTRA AI / OpenPilot AI API is online. UI index.html not yet built."}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
