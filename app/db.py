from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "app.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")
_IS_SQLITE = DATABASE_URL.startswith("sqlite")

# check_same_thread=False - планировщик (scheduler.py) опрашивает Dota2 и CS2
# параллельно из разных потоков ThreadPoolExecutor; timeout - это busy_timeout
# sqlite3, без него параллельная запись из двух job'ов сразу падает с
# "database is locked" вместо ожидания освобождения блокировки.
_connect_args = {"check_same_thread": False, "timeout": 30} if _IS_SQLITE else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)

if _IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
        # WAL - читатели (FastAPI/Streamlit) не блокируют писателя (планировщик),
        # и наоборот; busy_timeout продублирован здесь на уровне SQLite для
        # запросов, идущих в обход connect_args (например PRAGMA-соединений).
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()
        # pysqlite сам эмитирует implicit BEGIN не там, где нужно SQLAlchemy
        # (задокументированный баг, см. SQLAlchemy sqlite dialect notes:
        # "Serializable isolation / Savepoints / Transactional DDL") - из-за
        # этого busy_timeout выше не всегда успевал сработать и параллельная
        # запись двух job'ов планировщика падала с "database is locked"
        # почти мгновенно вместо ожидания. Отключаем встроенное управление
        # транзакциями pysqlite и передаём его SQLAlchemy явно (событие
        # "begin" ниже) - тогда busy_timeout применяется предсказуемо.
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _sqlite_begin(conn) -> None:
        conn.exec_driver_sql("BEGIN")


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()
