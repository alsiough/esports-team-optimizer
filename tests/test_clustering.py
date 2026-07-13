"""Тесты clustering.py - граничный случай "недостаточно данных для кластеризации" (ТЗ 10)
и базовая проверка, что KMeans действительно разделяет явно разные группы игроков.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.clustering import compute_clusters, latest_clusters
from app.models import Player, PlayerSnapshot


def _add_player(session, *, external_id, metrics) -> Player:
    player = Player(game="cs2", external_id=external_id, nickname=external_id, role=None)
    session.add(player)
    session.flush()
    session.add(
        PlayerSnapshot(player_id=player.id, taken_at=dt.datetime.now(dt.timezone.utc), metrics=metrics)
    )
    session.commit()
    return player


def test_compute_clusters_no_players_returns_zero(db_session):
    assert compute_clusters(db_session, "cs2") == 0


def test_compute_clusters_single_player_returns_zero(db_session):
    _add_player(db_session, external_id="only-one", metrics={"winrate": 0.5, "kd": 1.0, "adr": 70, "hs_pct": 0.4})
    assert compute_clusters(db_session, "cs2") == 0


def test_compute_clusters_unknown_game_raises(db_session):
    with pytest.raises(ValueError):
        compute_clusters(db_session, "lol")


def test_compute_clusters_separates_two_distinct_groups(db_session):
    # небольшой джиттер на игрока - чтобы точки внутри группы не совпадали
    # побитово (иначе sklearn ругается ConvergenceWarning на дубликаты точек)
    def jitter(base: dict, i: int) -> dict:
        return {k: v + i * 0.01 for k, v in base.items()}

    strong = {"winrate": 0.8, "kd": 1.8, "adr": 100.0, "hs_pct": 0.6}
    weak = {"winrate": 0.2, "kd": 0.5, "adr": 40.0, "hs_pct": 0.2}
    strong_players = [_add_player(db_session, external_id=f"strong-{i}", metrics=jitter(strong, i)) for i in range(4)]
    weak_players = [_add_player(db_session, external_id=f"weak-{i}", metrics=jitter(weak, i)) for i in range(4)]

    clustered = compute_clusters(db_session, "cs2")
    assert clustered == 8

    labels = latest_clusters(db_session, "cs2")
    strong_labels = {labels[p.id] for p in strong_players}
    weak_labels = {labels[p.id] for p in weak_players}
    assert len(strong_labels) == 1
    assert len(weak_labels) == 1
    assert strong_labels != weak_labels
