"""FastAPI backend поверх БД - см. TECHNICAL_SPEC.md 5.7.

Минимальный REST-слой для дашборда: чтение игроков/истории/кластеров,
запуск оптимизатора состава и внеочередного опроса источников. Планировщик
(app/scheduler.py) стартует на startup-хуке приложения - до этого он только
собирался, но не запускался (см. app/scheduler.py:create_scheduler).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.clustering import compute_clusters, latest_cluster_rows, latest_clusters
from app.collectors.faceit import FaceitCollector
from app.db import get_session, init_db
from app.models import Player, PlayerSnapshot, Rating
from app.optimizer import (
    DEFAULT_ACTIVE_DAYS,
    DEFAULT_MAX_PER_CLUSTER,
    DEFAULT_MIN_CLUSTERS,
    TEAM_SIZE,
    optimize_team,
)
from app.rating import compute_ratings, latest_ratings
from app.scheduler import (
    CS2_POOL_LIMIT,
    CS2_REGION,
    DOTA2_POOL_LIMIT,
    create_scheduler,
    ingest_cs2,
    ingest_dota2,
)

logger = logging.getLogger(__name__)

GAMES = ("dota2", "cs2")

_scheduler: BackgroundScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _scheduler
    init_db()
    _scheduler = create_scheduler()
    _scheduler.start()
    logger.info("планировщик запущен")
    try:
        yield
    finally:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)


app = FastAPI(title="Esports Team Optimizer", lifespan=lifespan)


def get_db() -> Session:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _check_game(game: str) -> None:
    if game not in GAMES:
        raise HTTPException(status_code=400, detail=f"неизвестная игра: {game}")


class PlayerBase(BaseModel):
    id: int
    game: str
    external_id: str
    nickname: str
    team: str | None
    role: str | None
    first_seen: dt.datetime
    last_seen: dt.datetime


class PlayerOut(PlayerBase):
    rating: float | None = None
    cluster_label: int | None = None


def _player_base(player: Player) -> PlayerBase:
    return PlayerBase(
        id=player.id,
        game=player.game,
        external_id=player.external_id,
        nickname=player.nickname,
        team=player.team,
        role=player.role,
        first_seen=player.first_seen,
        last_seen=player.last_seen,
    )


def _player_out(player: Player, rating: float | None, cluster_label: int | None) -> PlayerOut:
    return PlayerOut(**_player_base(player).model_dump(), rating=rating, cluster_label=cluster_label)


class SnapshotOut(BaseModel):
    taken_at: dt.datetime
    metrics: dict


class RatingOut(BaseModel):
    computed_at: dt.datetime
    rating_value: float
    rating_version: str


class PlayerHistoryOut(BaseModel):
    player: PlayerBase
    snapshots: list[SnapshotOut]
    ratings: list[RatingOut]


class ClusterOut(BaseModel):
    player_id: int
    nickname: str
    algorithm: str
    cluster_label: int
    computed_at: dt.datetime


class TeamOptimizeRequest(BaseModel):
    game: str
    team_size: int = TEAM_SIZE
    max_per_cluster: int = DEFAULT_MAX_PER_CLUSTER
    min_clusters: int = DEFAULT_MIN_CLUSTERS
    active_days: int = DEFAULT_ACTIVE_DAYS


class TeamMemberOut(BaseModel):
    player_id: int
    nickname: str
    rating: float
    role: str | None
    cluster_label: int | None


class TeamOptimizeResponse(BaseModel):
    feasible: bool
    message: str
    objective_value: float | None
    constraints: dict
    team: list[TeamMemberOut]


class RefreshRequest(BaseModel):
    game: str | None = None  # None - опросить оба источника


class RefreshResult(BaseModel):
    game: str
    snapshots_stored: int
    players_rated: int
    players_clustered: int


class RefreshResponse(BaseModel):
    results: list[RefreshResult]


@app.get("/players", response_model=list[PlayerOut])
def list_players(
    game: str = Query(..., description="dota2 | cs2"),
    cluster: int | None = Query(None, description="фильтр по метке кластера"),
    min_rating: float | None = Query(None),
    max_rating: float | None = Query(None),
    db: Session = Depends(get_db),
) -> list[PlayerOut]:
    _check_game(game)
    ratings = latest_ratings(db, game)
    clusters = latest_clusters(db, game)

    result = []
    for player in db.query(Player).filter(Player.game == game).order_by(Player.nickname).all():
        rating = ratings.get(player.id)
        cluster_label = clusters.get(player.id)
        if cluster is not None and cluster_label != cluster:
            continue
        if min_rating is not None and (rating is None or rating < min_rating):
            continue
        if max_rating is not None and (rating is None or rating > max_rating):
            continue
        result.append(_player_out(player, rating, cluster_label))
    return result


@app.get("/players/{player_id}/history", response_model=PlayerHistoryOut)
def player_history(player_id: int, db: Session = Depends(get_db)) -> PlayerHistoryOut:
    player = db.get(Player, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="игрок не найден")

    snapshots = (
        db.query(PlayerSnapshot)
        .filter(PlayerSnapshot.player_id == player_id)
        .order_by(PlayerSnapshot.taken_at)
        .all()
    )
    ratings = db.query(Rating).filter(Rating.player_id == player_id).order_by(Rating.computed_at).all()

    return PlayerHistoryOut(
        player=_player_base(player),
        snapshots=[SnapshotOut(taken_at=s.taken_at, metrics=s.metrics) for s in snapshots],
        ratings=[
            RatingOut(computed_at=r.computed_at, rating_value=r.rating_value, rating_version=r.rating_version)
            for r in ratings
        ],
    )


@app.get("/clusters", response_model=list[ClusterOut])
def list_clusters(game: str = Query(..., description="dota2 | cs2"), db: Session = Depends(get_db)) -> list[ClusterOut]:
    _check_game(game)
    nicknames = {p.id: p.nickname for p in db.query(Player).filter(Player.game == game).all()}
    return [
        ClusterOut(
            player_id=c.player_id,
            nickname=nicknames.get(c.player_id, ""),
            algorithm=c.algorithm,
            cluster_label=c.cluster_label,
            computed_at=c.computed_at,
        )
        for c in latest_cluster_rows(db, game)
    ]


@app.post("/team/optimize", response_model=TeamOptimizeResponse)
def team_optimize(payload: TeamOptimizeRequest, db: Session = Depends(get_db)) -> TeamOptimizeResponse:
    _check_game(payload.game)
    result = optimize_team(
        db,
        payload.game,
        team_size=payload.team_size,
        max_per_cluster=payload.max_per_cluster,
        min_clusters=payload.min_clusters,
        active_days=payload.active_days,
    )

    team: list[TeamMemberOut] = []
    if result.feasible:
        ratings = latest_ratings(db, payload.game)
        clusters = latest_clusters(db, payload.game) if payload.game == "cs2" else {}
        for player_id in result.player_ids:
            player = db.get(Player, player_id)
            team.append(
                TeamMemberOut(
                    player_id=player_id,
                    nickname=player.nickname,
                    rating=ratings.get(player_id, 0.0),
                    role=player.role,
                    cluster_label=clusters.get(player_id),
                )
            )

    return TeamOptimizeResponse(
        feasible=result.feasible,
        message=result.message,
        objective_value=result.objective_value,
        constraints=result.constraints,
        team=team,
    )


def _refresh_one(db: Session, game: str) -> RefreshResult:
    if game == "dota2":
        stored = ingest_dota2(db, pool_limit=DOTA2_POOL_LIMIT)
    else:
        api_key = os.getenv("FACEIT_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="cs2: FACEIT_API_KEY не задан")
        stored = ingest_cs2(db, FaceitCollector(api_key=api_key, region=CS2_REGION), pool_limit=CS2_POOL_LIMIT)

    rated = compute_ratings(db, game) if stored else 0
    # Кластеризация не завязана на планировщик (см. app/scheduler.py), но
    # /refresh должен готовить данные к демонстрации целиком - без неё CS2
    # оптимизатор (нужен кластер каждого игрока) остался бы на устаревших данных.
    clustered = compute_clusters(db, game) if stored else 0
    return RefreshResult(game=game, snapshots_stored=stored, players_rated=rated, players_clustered=clustered)


@app.post("/refresh", response_model=RefreshResponse)
def refresh(payload: RefreshRequest = RefreshRequest(), db: Session = Depends(get_db)) -> RefreshResponse:
    games = [payload.game] if payload.game is not None else list(GAMES)
    for game in games:
        _check_game(game)
    return RefreshResponse(results=[_refresh_one(db, game) for game in games])
