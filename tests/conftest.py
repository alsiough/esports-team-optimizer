"""Общие фикстуры: изолированная in-memory SQLite БД для тестов.

НЕ трогает рабочую data/app.db и не требует переменной DATABASE_URL - модули
rating.py/clustering.py/optimizer.py принимают Session как параметр и не
завязаны на глобальный engine из app.db.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base


@pytest.fixture
def db_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
