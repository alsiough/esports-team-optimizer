"""Подбор оптимального состава из 5 игроков методом целочисленного
программирования (PuLP). См. TECHNICAL_SPEC.md 5.5.

Модуль разбит на два слоя:
- optimize_candidates() - чистая функция над готовым списком кандидатов, без
  обращения к БД - удобно проверять на синтетических данных.
- optimize_team() - достаёт кандидатов (последний рейтинг/снапшот/кластер) из
  БД, вызывает optimize_candidates() и сохраняет результат в team_proposals.

Инфизибл-ограничения не приводят к исключению - возвращается
OptimizationResult(feasible=False, message=...) с объяснением (ТЗ, критерии
приёмки, раздел 10).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pulp
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Cluster, Player, PlayerSnapshot, Rating, TeamProposal

TEAM_SIZE = 5
DEFAULT_MAX_PER_CLUSTER = 2
DEFAULT_MIN_CLUSTERS = 3
DEFAULT_ACTIVE_DAYS = 30
DOTA2_ROLES = [str(r) for r in range(1, 6)]


@dataclass
class Candidate:
    player_id: int
    role: str | None  # Dota 2: "1".."5"; CS2: всегда None (роль = кластер)
    cluster_label: int | None  # CS2: последняя метка кластера; Dota 2 не используется
    rating: float


@dataclass
class OptimizationResult:
    feasible: bool
    player_ids: list[int] = field(default_factory=list)
    objective_value: float | None = None
    message: str = ""
    constraints: dict = field(default_factory=dict)


def optimize_candidates(
    candidates: list[Candidate],
    game: str,
    *,
    team_size: int = TEAM_SIZE,
    max_per_cluster: int = DEFAULT_MAX_PER_CLUSTER,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
) -> OptimizationResult:
    """Решает задачу выбора состава (ЦП, PuLP) над готовым списком кандидатов одной игры."""
    if game not in ("dota2", "cs2"):
        raise ValueError(f"неизвестная игра: {game}")

    constraints = {"team_size": team_size}
    if game == "cs2":
        constraints.update(max_per_cluster=max_per_cluster, min_clusters=min_clusters)

    if len(candidates) < team_size:
        return OptimizationResult(
            feasible=False,
            message=f"недостаточно активных кандидатов: {len(candidates)} < {team_size}",
            constraints=constraints,
        )

    problem = pulp.LpProblem("team_optimization", pulp.LpMaximize)
    x = {c.player_id: pulp.LpVariable(f"x_{c.player_id}", cat=pulp.LpBinary) for c in candidates}

    problem += pulp.lpSum(c.rating * x[c.player_id] for c in candidates)
    problem += pulp.lpSum(x.values()) == team_size

    if game == "dota2":
        # Ровно один игрок на каждую роль 1-5 (кандидаты без роли уже
        # отфильтрованы на этапе сбора кандидатов - см. _gather_candidates).
        for role in DOTA2_ROLES:
            role_candidates = [c for c in candidates if c.role == role]
            problem += pulp.lpSum(x[c.player_id] for c in role_candidates) == 1
    else:  # cs2
        # Жёстких ролей нет - разнообразие стилей обеспечивается лимитом на
        # кластер (MAX_PER_CLUSTER) и минимальным числом разных кластеров
        # (MIN_CLUSTERS) в составе.
        cluster_labels = sorted({c.cluster_label for c in candidates})
        cluster_used = {}
        for label in cluster_labels:
            cluster_candidates = [c for c in candidates if c.cluster_label == label]
            cluster_sum = pulp.lpSum(x[c.player_id] for c in cluster_candidates)
            problem += cluster_sum <= max_per_cluster
            used = pulp.LpVariable(f"cluster_used_{label}", cat=pulp.LpBinary)
            # used == 1 тогда и только тогда, когда из кластера выбран хоть один игрок
            problem += cluster_sum >= used
            problem += cluster_sum <= used * max_per_cluster
            cluster_used[label] = used
        problem += pulp.lpSum(cluster_used.values()) >= min_clusters

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[status] != "Optimal":
        return OptimizationResult(
            feasible=False,
            message=f"не удалось собрать состав под заданные ограничения: {pulp.LpStatus[status]}",
            constraints=constraints,
        )

    selected_ids = [c.player_id for c in candidates if pulp.value(x[c.player_id]) > 0.5]
    return OptimizationResult(
        feasible=True,
        player_ids=selected_ids,
        objective_value=float(pulp.value(problem.objective)),
        constraints=constraints,
    )


def _active_player_ids(session: Session, game: str, cutoff: dt.datetime) -> set[int]:
    """player_id игроков, чей последний снапшот не старше cutoff.

    Сравнение с cutoff делается на уровне SQL (как в clustering.py), а не в
    Python - SQLite отдаёт taken_at naive datetime, и сравнение с aware
    datetime в Python упало бы с TypeError.
    """
    latest = (
        session.query(PlayerSnapshot.player_id, func.max(PlayerSnapshot.taken_at).label("latest_taken_at"))
        .join(Player)
        .filter(Player.game == game)
        .group_by(PlayerSnapshot.player_id)
        .subquery()
    )
    rows = session.query(latest.c.player_id).filter(latest.c.latest_taken_at >= cutoff).all()
    return {player_id for (player_id,) in rows}


def _latest_ratings(session: Session, game: str) -> dict[int, float]:
    latest = (
        session.query(Rating.player_id, func.max(Rating.computed_at).label("latest_computed_at"))
        .join(Player)
        .filter(Player.game == game)
        .group_by(Rating.player_id)
        .subquery()
    )
    rows = (
        session.query(Rating.player_id, Rating.rating_value)
        .join(
            latest,
            (latest.c.player_id == Rating.player_id) & (latest.c.latest_computed_at == Rating.computed_at),
        )
        .all()
    )
    return dict(rows)


def _latest_clusters(session: Session, game: str) -> dict[int, int]:
    latest = (
        session.query(Cluster.player_id, func.max(Cluster.computed_at).label("latest_computed_at"))
        .join(Player)
        .filter(Player.game == game)
        .group_by(Cluster.player_id)
        .subquery()
    )
    rows = (
        session.query(Cluster.player_id, Cluster.cluster_label)
        .join(
            latest,
            (latest.c.player_id == Cluster.player_id) & (latest.c.latest_computed_at == Cluster.computed_at),
        )
        .all()
    )
    return dict(rows)


def _gather_candidates(session: Session, game: str, active_days: int) -> list[Candidate]:
    """Активные кандидаты игры: снапшот не старше active_days, есть посчитанный
    рейтинг, для Dota 2 - определена роль, для CS2 - есть кластер (ТЗ 5.5)."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=active_days)
    active_ids = _active_player_ids(session, game, cutoff)
    ratings = _latest_ratings(session, game)
    clusters = _latest_clusters(session, game) if game == "cs2" else {}

    candidates = []
    for player in session.query(Player).filter(Player.game == game).all():
        if player.id not in active_ids:
            continue
        rating = ratings.get(player.id)
        if rating is None:
            continue
        if game == "dota2":
            if player.role not in DOTA2_ROLES:
                continue
            cluster_label = None
        else:
            cluster_label = clusters.get(player.id)
            if cluster_label is None:
                continue
        candidates.append(
            Candidate(player_id=player.id, role=player.role, cluster_label=cluster_label, rating=rating)
        )
    return candidates


def optimize_team(
    session: Session,
    game: str,
    *,
    team_size: int = TEAM_SIZE,
    max_per_cluster: int = DEFAULT_MAX_PER_CLUSTER,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    active_days: int = DEFAULT_ACTIVE_DAYS,
) -> OptimizationResult:
    """Подбирает состав из 5 игроков для game и сохраняет результат в team_proposals.

    При невозможности собрать состав под заданные ограничения возвращает
    OptimizationResult(feasible=False, message=...) вместо исключения -
    ничего не пишет в team_proposals в этом случае.
    """
    if game not in ("dota2", "cs2"):
        raise ValueError(f"неизвестная игра: {game}")

    candidates = _gather_candidates(session, game, active_days)
    result = optimize_candidates(
        candidates,
        game,
        team_size=team_size,
        max_per_cluster=max_per_cluster,
        min_clusters=min_clusters,
    )
    result.constraints["active_days"] = active_days

    if result.feasible:
        session.add(
            TeamProposal(
                game=game,
                constraints=result.constraints,
                player_ids=result.player_ids,
                objective_value=result.objective_value,
            )
        )
        session.commit()

    return result
