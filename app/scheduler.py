from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Callable

import httpx
from sqlalchemy.orm import Session

from app.collectors.base import BaseCollector, SourceUnavailableError
from app.collectors.faceit import FaceitCollector
from app.collectors.opendota import OpenDotaCollector
from app.models import Player, PlayerSnapshot
from app.normalize import (
    normalize_faceit_player,
    normalize_faceit_snapshot,
    normalize_opendota_player,
    normalize_opendota_snapshot,
)

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
    return ingest(
        session,
        game="dota2",
        collector=collector or OpenDotaCollector(),
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
