"""Database Connection and Session Factory for SQLite."""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Database path in ai-system directory
DB_PATH = Path(__file__).resolve().parent.parent.parent / "ai_system.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Connect with check_same_thread=False for multi-threaded FastAPI / asyncio workers
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    """Creates database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
