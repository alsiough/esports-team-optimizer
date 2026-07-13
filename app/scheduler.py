from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Any, Callable

import httpx
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session

from app.collectors.base import BaseCollector, SourceUnavailableError
from app.collectors.faceit import FaceitCollector
from app.collectors.opendota import OpenDotaCollector
from app.db import get_session
from app.models import Player, PlayerSnapshot
from app.normalize import (
    normalize_faceit_player,
    normalize_faceit_snapshot,
    normalize_opendota_player,
    normalize_opendota_snapshot,
)
from app.rating import compute_ratings

logger = logging.getLogger(__name__)

NormalizePlayerFn = Callable[[dict[str, Any]], dict[str, Any] | None]
NormalizeSnapshotFn = Callable[[dict[str, Any]], dict[str, Any]]


def _get_or_create_player(
    session: Session, *, game: str, external_id: str, nickname: str, team: str | None
) -> Player:
    player = session.query(Player).filter_by(game=game, external_id=external_id).one_or_none()
    now = dt.datetime.now(dt.timezone.utc)
    if player is None:
        player = Player(game=game, external_id=external_id, nickname=nickname, team=team)
        session.add(player)
    else:
        player.nickname = nickname
        player.team = team
        player.last_seen = now
    return player


def ingest(
    session: Session,
    *,
    game: str,
    collector: BaseCollector,
    normalize_player: NormalizePlayerFn,
    normalize_snapshot: NormalizeSnapshotFn,
    pool_limit: int | None = None,
) -> int:
    """Опрашивает источник и пишет по одному новому снапшоту на игрока. Возвращает число сохранённых снапшотов."""
    try:
        pool = collector.fetch_player_pool()
    except SourceUnavailableError as exc:
        logger.error("%s: пул игроков недоступен: %s", game, exc)
        return 0

    if pool_limit is not None:
        pool = pool[:pool_limit]

    stored = 0
    for raw_entry in pool:
        identity = normalize_player(raw_entry)
        if identity is None:
            continue

        try:
            raw_stats = collector.fetch_player_stats(identity["external_id"])
        except (SourceUnavailableError, httpx.HTTPError) as exc:
            logger.warning("%s: пропуск игрока %s: %s", game, identity["external_id"], exc)
            continue

        snapshot = normalize_snapshot(raw_stats)
        player = _get_or_create_player(
            session,
            game=game,
            external_id=identity["external_id"],
            nickname=identity["nickname"],
            team=identity["team"],
        )
        if snapshot["role"] is not None:
            player.role = str(snapshot["role"])
        session.flush()

        session.add(
            PlayerSnapshot(
                player_id=player.id,
                taken_at=dt.datetime.now(dt.timezone.utc),
                metrics=snapshot["metrics"],
            )
        )
        stored += 1

    session.commit()
    return stored


def ingest_dota2(session: Session, collector: OpenDotaCollector | None = None, pool_limit: int | None = None) -> int:
    # OPENDOTA_API_KEY опционален (в отличие от FACEIT_API_KEY) - публичный
    # лимит и без ключа работает, ключ только поднимает лимиты (см. CLAUDE.md)
    return ingest(
        session,
        game="dota2",
        collector=collector or OpenDotaCollector(api_key=os.getenv("OPENDOTA_API_KEY")),
        normalize_player=normalize_opendota_player,
        normalize_snapshot=normalize_opendota_snapshot,
        pool_limit=pool_limit,
    )


def ingest_cs2(session: Session, collector: FaceitCollector, pool_limit: int | None = None) -> int:
    return ingest(
        session,
        game="cs2",
        collector=collector,
        normalize_player=normalize_faceit_player,
        normalize_snapshot=normalize_faceit_snapshot,
        pool_limit=pool_limit,
    )


# Бюджет OpenDota - 60 запросов/мин и 50 000/мес; при пуле в тысячи про-игроков
# опрашивать его целиком каждый цикл нельзя (см. TECHNICAL_SPEC.md 11), поэтому
# планировщик берёт ограниченный срез пула за один запуск. Интервалы и размеры
# срезов - конфигурируемые, чтобы их можно было подстроить без правки кода.
DOTA2_POLL_INTERVAL_MINUTES = int(os.getenv("DOTA2_POLL_INTERVAL_MINUTES", "60"))
DOTA2_POOL_LIMIT = int(os.getenv("DOTA2_POOL_LIMIT", "20"))
CS2_POLL_INTERVAL_MINUTES = int(os.getenv("CS2_POLL_INTERVAL_MINUTES", "60"))
CS2_POOL_LIMIT = int(os.getenv("CS2_POOL_LIMIT", "50"))
CS2_REGION = os.getenv("FACEIT_REGION", "EU")


def _job_ingest_dota2() -> None:
    session = get_session()
    try:
        stored = ingest_dota2(session, pool_limit=DOTA2_POOL_LIMIT)
        logger.info("dota2: job завершён, снапшотов сохранено: %s", stored)
        if stored:
            rated = compute_ratings(session, "dota2")
            logger.info("dota2: рейтинг пересчитан для %s игроков", rated)
    except Exception:
        logger.exception("dota2: job упал с ошибкой")
    finally:
        session.close()


def _job_ingest_cs2() -> None:
    api_key = os.getenv("FACEIT_API_KEY")
    if not api_key:
        logger.warning("cs2: FACEIT_API_KEY не задан, job пропущен")
        return
    session = get_session()
    try:
        collector = FaceitCollector(api_key=api_key, region=CS2_REGION)
        stored = ingest_cs2(session, collector, pool_limit=CS2_POOL_LIMIT)
        logger.info("cs2: job завершён, снапшотов сохранено: %s", stored)
        if stored:
            rated = compute_ratings(session, "cs2")
            logger.info("cs2: рейтинг пересчитан для %s игроков", rated)
    except Exception:
        logger.exception("cs2: job упал с ошибкой")
    finally:
        session.close()


def create_scheduler() -> BackgroundScheduler:
    """Собирает BackgroundScheduler с job'ами опроса обоих источников. Не запускает - вызывающий код сам решает, когда start()/shutdown()."""
    scheduler = BackgroundScheduler(
        timezone="UTC",
        executors={"default": ThreadPoolExecutor(max_workers=2)},
    )
    now = dt.datetime.now(dt.timezone.utc)
    scheduler.add_job(
        _job_ingest_dota2,
        trigger=IntervalTrigger(minutes=DOTA2_POLL_INTERVAL_MINUTES),
        id="ingest_dota2",
        next_run_time=now,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _job_ingest_cs2,
        trigger=IntervalTrigger(minutes=CS2_POLL_INTERVAL_MINUTES),
        id="ingest_cs2",
        next_run_time=now,
        max_instances=1,
        coalesce=True,
    )
    return scheduler
