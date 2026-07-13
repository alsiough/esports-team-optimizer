"""Тесты optimizer.py - ЦП-подбор состава (ТЗ 5.5).

Два уровня: optimize_candidates() на синтетических кандидатах (без БД) и
optimize_team() поверх in-memory SQLite (db_session) - проверяет фильтрацию
активных кандидатов и запись team_proposals. Граничный случай "инфизибл-
ограничения" (ТЗ 10) покрыт на обоих уровнях.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.models import Cluster, Player, PlayerSnapshot, Rating, TeamProposal
from app.optimizer import Candidate, optimize_candidates, optimize_team


# --- optimize_candidates(): синтетические кандидаты, без БД ---


class TestOptimizeCandidatesDota2:
    def _candidates(self) -> list[Candidate]:
        return [
            Candidate(player_id=1, role="1", cluster_label=None, rating=5.0),
            Candidate(player_id=2, role="1", cluster_label=None, rating=1.0),
            Candidate(player_id=3, role="2", cluster_label=None, rating=2.0),
            Candidate(player_id=4, role="3", cluster_label=None, rating=2.0),
            Candidate(player_id=5, role="4", cluster_label=None, rating=2.0),
            Candidate(player_id=6, role="5", cluster_label=None, rating=2.0),
        ]

    def test_picks_best_candidate_per_role(self):
        result = optimize_candidates(self._candidates(), "dota2")
        assert result.feasible
        assert sorted(result.player_ids) == [1, 3, 4, 5, 6]
        assert result.objective_value == pytest.approx(13.0)

    def test_missing_role_is_infeasible(self):
        # роль "1" отсутствует вовсе, но кандидатов всё равно >= team_size -
        # проверяем именно ограничение "один игрок на роль", а не нехватку кандидатов
        candidates = [c for c in self._candidates() if c.role != "1"]
        candidates.append(Candidate(player_id=99, role="2", cluster_label=None, rating=0.5))
        assert len(candidates) == 5

        result = optimize_candidates(candidates, "dota2")
        assert not result.feasible
        assert "не удалось" in result.message

    def test_too_few_candidates_is_infeasible(self):
        result = optimize_candidates(self._candidates()[:3], "dota2")
        assert not result.feasible
        assert "недостаточно" in result.message


class TestOptimizeCandidatesCs2:
    def _candidates(self) -> list[Candidate]:
        return [
            Candidate(player_id=1, role=None, cluster_label=0, rating=10.0),
            Candidate(player_id=2, role=None, cluster_label=0, rating=9.0),
            Candidate(player_id=3, role=None, cluster_label=0, rating=8.0),
            Candidate(player_id=4, role=None, cluster_label=1, rating=3.0),
            Candidate(player_id=5, role=None, cluster_label=1, rating=2.0),
            Candidate(player_id=6, role=None, cluster_label=2, rating=1.0),
            Candidate(player_id=7, role=None, cluster_label=2, rating=0.5),
        ]

    def test_respects_max_per_cluster_and_min_clusters(self):
        result = optimize_candidates(self._candidates(), "cs2")
        assert result.feasible
        assert sorted(result.player_ids) == [1, 2, 4, 5, 6]
        assert result.objective_value == pytest.approx(25.0)

    def test_min_clusters_unreachable_is_infeasible(self):
        candidates = [c for c in self._candidates() if c.cluster_label != 2]  # только 2 кластера
        result = optimize_candidates(candidates, "cs2")
        assert not result.feasible

    def test_max_per_cluster_too_small_is_infeasible(self):
        candidates = [Candidate(player_id=i, role=None, cluster_label=0, rating=float(i)) for i in range(1, 8)]
        result = optimize_candidates(candidates, "cs2", max_per_cluster=2, min_clusters=1)
        assert not result.feasible  # team_size=5, но один кластер и лимит 2


def test_optimize_candidates_unknown_game_raises():
    with pytest.raises(ValueError):
        optimize_candidates([], "lol")


# --- optimize_team(): интеграция с БД (in-memory SQLite через db_session) ---


def _add_dota2_player(session, *, external_id, nickname, role, rating, snapshot_age_days=1) -> Player:
    now = dt.datetime.now(dt.timezone.utc)
    player = Player(game="dota2", external_id=external_id, nickname=nickname, role=role)
    session.add(player)
    session.flush()
    session.add(
        PlayerSnapshot(
            player_id=player.id,
            taken_at=now - dt.timedelta(days=snapshot_age_days),
            metrics={"winrate": 0.5, "kda": 3.0, "gpm": 500, "xpm": 500, "hero_damage": 10000},
        )
    )
    session.add(Rating(player_id=player.id, rating_value=rating, rating_version="v1", computed_at=now))
    session.commit()
    return player


def _add_cs2_player(session, *, external_id, nickname, cluster_label, rating, snapshot_age_days=1) -> Player:
    now = dt.datetime.now(dt.timezone.utc)
    player = Player(game="cs2", external_id=external_id, nickname=nickname, role=None)
    session.add(player)
    session.flush()
    session.add(
        PlayerSnapshot(
            player_id=player.id,
            taken_at=now - dt.timedelta(days=snapshot_age_days),
            metrics={"winrate": 0.5, "kd": 1.0, "adr": 70, "hs_pct": 0.4},
        )
    )
    session.add(Rating(player_id=player.id, rating_value=rating, rating_version="v1", computed_at=now))
    session.add(Cluster(player_id=player.id, algorithm="kmeans_k3", cluster_label=cluster_label, computed_at=now))
    session.commit()
    return player


class TestOptimizeTeamDota2:
    def test_builds_valid_team_and_saves_proposal(self, db_session):
        for role, rating in [("1", 5.0), ("2", 4.0), ("3", 3.0), ("4", 2.0), ("5", 1.0)]:
            _add_dota2_player(db_session, external_id=f"ext-{role}", nickname=f"p{role}", role=role, rating=rating)

        result = optimize_team(db_session, "dota2")
        assert result.feasible
        assert len(result.player_ids) == 5
        assert result.objective_value == pytest.approx(15.0)
        assert db_session.query(TeamProposal).count() == 1

    def test_stale_snapshot_excluded_from_candidates(self, db_session):
        # роль "1": свежий кандидат с низким рейтингом vs устаревший (>30 дней) с высоким рейтингом
        fresh = _add_dota2_player(db_session, external_id="fresh", nickname="Fresh", role="1", rating=1.0)
        _add_dota2_player(
            db_session, external_id="stale", nickname="Stale", role="1", rating=10.0, snapshot_age_days=40
        )
        for role, rating in [("2", 4.0), ("3", 3.0), ("4", 2.0), ("5", 1.0)]:
            _add_dota2_player(db_session, external_id=f"ext-{role}", nickname=f"p{role}", role=role, rating=rating)

        result = optimize_team(db_session, "dota2", active_days=30)
        assert result.feasible
        assert fresh.id in result.player_ids

    def test_missing_role_coverage_is_infeasible_and_not_saved(self, db_session):
        for role, rating in [("1", 5.0), ("2", 4.0), ("3", 3.0), ("4", 2.0)]:  # роль "5" отсутствует
            _add_dota2_player(db_session, external_id=f"ext-{role}", nickname=f"p{role}", role=role, rating=rating)

        result = optimize_team(db_session, "dota2")
        assert not result.feasible
        assert db_session.query(TeamProposal).count() == 0


class TestOptimizeTeamCs2:
    def test_builds_valid_team_with_cluster_diversity(self, db_session):
        ratings_by_cluster = {0: [10.0, 9.0, 8.0], 1: [3.0, 2.0], 2: [1.0, 0.5]}
        for cluster_label, ratings in ratings_by_cluster.items():
            for i, rating in enumerate(ratings):
                _add_cs2_player(
                    db_session,
                    external_id=f"c{cluster_label}-{i}",
                    nickname=f"c{cluster_label}p{i}",
                    cluster_label=cluster_label,
                    rating=rating,
                )

        result = optimize_team(db_session, "cs2")
        assert result.feasible
        assert len(result.player_ids) == 5
        assert db_session.query(TeamProposal).count() == 1

    def test_player_without_cluster_excluded(self, db_session):
        best = _add_cs2_player(db_session, external_id="no-cluster", nickname="Best", cluster_label=0, rating=100.0)
        # у "Best" искусственно уберём кластер, имитируя игрока без пересчитанной кластеризации
        db_session.query(Cluster).filter_by(player_id=best.id).delete()
        db_session.commit()

        for cluster_label, ratings in {0: [5.0, 4.0], 1: [3.0], 2: [2.0], 3: [1.0]}.items():
            for i, rating in enumerate(ratings):
                _add_cs2_player(
                    db_session,
                    external_id=f"c{cluster_label}-{i}",
                    nickname=f"c{cluster_label}p{i}",
                    cluster_label=cluster_label,
                    rating=rating,
                )

        result = optimize_team(db_session, "cs2")
        assert result.feasible
        assert best.id not in result.player_ids

    def test_min_clusters_unreachable_is_infeasible_and_not_saved(self, db_session):
        for cluster_label in (0, 1):
            for i in range(3):
                _add_cs2_player(
                    db_session,
                    external_id=f"c{cluster_label}-{i}",
                    nickname=f"c{cluster_label}p{i}",
                    cluster_label=cluster_label,
                    rating=float(i),
                )

        result = optimize_team(db_session, "cs2", min_clusters=3)
        assert not result.feasible
        assert db_session.query(TeamProposal).count() == 0


def test_optimize_team_unknown_game_raises(db_session):
    with pytest.raises(ValueError):
        optimize_team(db_session, "lol")
