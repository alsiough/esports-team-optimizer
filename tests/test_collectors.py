"""Тесты collectors/* - HTTP-адаптеры на моках (httpx.MockTransport), без реальной сети.

Покрывает: успешные ответы, неожиданный формат ответа, backoff/retry на
429/5xx (см. app/collectors/base.py), исчерпание попыток и граничный случай
"пустой пул" (ТЗ 10 - критерии приёмки/тестирование).
"""

from __future__ import annotations

import logging

import httpx
import pytest

import app.scheduler as scheduler
from app.collectors.base import SourceUnavailableError
from app.collectors.faceit import FaceitCollector
from app.collectors.opendota import OpenDotaCollector
from app.models import Player, PlayerSnapshot
from app.scheduler import BELOW_POOL_THRESHOLD_KEY, ingest, ingest_dota2


def _install_mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Подменяет httpx.Client так, чтобы использовался MockTransport - без реальной сети."""
    original_client_cls = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client_cls(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", fake_client)


class TestOpenDotaCollector:
    def test_fetch_player_pool_success(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/proPlayers"
            return httpx.Response(200, json=[{"account_id": 1, "name": "Foo"}])

        _install_mock_transport(monkeypatch, handler)
        assert OpenDotaCollector().fetch_player_pool() == [{"account_id": 1, "name": "Foo"}]

    def test_fetch_player_pool_empty(self, monkeypatch):
        _install_mock_transport(monkeypatch, lambda request: httpx.Response(200, json=[]))
        assert OpenDotaCollector().fetch_player_pool() == []

    def test_fetch_player_pool_bad_format_raises(self, monkeypatch):
        _install_mock_transport(monkeypatch, lambda request: httpx.Response(200, json={"unexpected": "dict"}))
        with pytest.raises(SourceUnavailableError):
            OpenDotaCollector().fetch_player_pool()

    def test_fetch_player_stats_combines_three_endpoints(self, monkeypatch):
        seen_paths = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            if request.url.path.endswith("/wl"):
                return httpx.Response(200, json={"win": 1, "lose": 1})
            return httpx.Response(200, json=[])

        _install_mock_transport(monkeypatch, handler)
        raw = OpenDotaCollector().fetch_player_stats("42")
        assert raw["account_id"] == "42"
        assert raw["wl"] == {"win": 1, "lose": 1}
        assert seen_paths == ["/api/players/42/wl", "/api/players/42/totals", "/api/players/42/matches"]

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("app.collectors.base.time.sleep", lambda _seconds: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(429) if calls["n"] < 3 else httpx.Response(200, json=[])

        _install_mock_transport(monkeypatch, handler)
        assert OpenDotaCollector().fetch_player_pool() == []
        assert calls["n"] == 3

    def test_exhausts_retries_raises_source_unavailable(self, monkeypatch):
        monkeypatch.setattr("app.collectors.base.time.sleep", lambda _seconds: None)
        _install_mock_transport(monkeypatch, lambda request: httpx.Response(500))
        with pytest.raises(SourceUnavailableError):
            OpenDotaCollector().fetch_player_pool()

    def test_retries_on_transport_error_then_raises(self, monkeypatch):
        monkeypatch.setattr("app.collectors.base.time.sleep", lambda _seconds: None)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(SourceUnavailableError):
            OpenDotaCollector().fetch_player_pool()


class TestFaceitCollector:
    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            FaceitCollector(api_key="")

    def test_fetch_player_pool_success_sends_bearer_token(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer secret-key"
            assert request.url.path == "/data/v4/rankings/games/cs2/regions/EU"
            return httpx.Response(200, json={"items": [{"player_id": "1", "nickname": "a"}]})

        _install_mock_transport(monkeypatch, handler)
        collector = FaceitCollector(api_key="secret-key")
        assert collector.fetch_player_pool() == [{"player_id": "1", "nickname": "a"}]

    def test_fetch_player_pool_missing_items_raises(self, monkeypatch):
        _install_mock_transport(monkeypatch, lambda request: httpx.Response(200, json={}))
        with pytest.raises(SourceUnavailableError):
            FaceitCollector(api_key="secret-key").fetch_player_pool()

    def test_fetch_player_stats_success(self, monkeypatch):
        _install_mock_transport(
            monkeypatch, lambda request: httpx.Response(200, json={"lifetime": {"Win Rate %": "50"}})
        )
        raw = FaceitCollector(api_key="secret-key").fetch_player_stats("player-1")
        assert raw["player_id"] == "player-1"
        assert raw["stats"] == {"lifetime": {"Win Rate %": "50"}}

    def test_fetch_player_pool_merges_multiple_regions_round_robin(self, monkeypatch):
        catalog = {
            "EU": [{"player_id": "eu1", "nickname": "a"}, {"player_id": "eu2", "nickname": "b"}],
            "NA": [{"player_id": "na1", "nickname": "c"}],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            region = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"items": catalog[region]})

        _install_mock_transport(monkeypatch, handler)
        collector = FaceitCollector(api_key="secret-key", regions=["EU", "NA"])
        pool = collector.fetch_player_pool()
        # вперемешку (round-robin), а не сначала весь EU, потом весь NA - иначе
        # обрезка пула до pool_limit в scheduler.ingest() забрала бы только EU
        assert [item["player_id"] for item in pool] == ["eu1", "na1", "eu2"]

    def test_fetch_player_pool_dedups_across_regions(self, monkeypatch):
        _install_mock_transport(
            monkeypatch, lambda request: httpx.Response(200, json={"items": [{"player_id": "dup", "nickname": "x"}]})
        )
        collector = FaceitCollector(api_key="secret-key", regions=["EU", "NA"])
        pool = collector.fetch_player_pool()
        assert pool == [{"player_id": "dup", "nickname": "x"}]


class _EmptyPoolCollector:
    game = "dota2"

    def fetch_player_pool(self):
        return []

    def fetch_player_stats(self, external_id):  # не должен вызываться для пустого пула
        raise AssertionError("fetch_player_stats не должен вызываться при пустом пуле")


class TestIngestEmptyPool:
    """ТЗ 10: пустой пул не должен приводить к падению pipeline опроса."""

    def test_ingest_empty_pool_returns_zero(self, db_session):
        stored = ingest(
            db_session,
            game="dota2",
            collector=_EmptyPoolCollector(),
            normalize_player=lambda raw: raw,
            normalize_snapshot=lambda raw: raw,
        )
        assert stored == 0


class _OnePlayerCollector:
    game = "dota2"

    def fetch_player_pool(self):
        return [{"external_id": "1", "nickname": "NoData", "team": None}]

    def fetch_player_stats(self, external_id):
        return {}  # источник не вернул данных по игроку


class TestIngestSkipsAllZeroSnapshot:
    """Снапшот, где все метрики нулевые ("нет данных источника" в normalize.py -
    см. _totals_avg/_winrate/normalize_faceit_snapshot), не должен попадать в БД:
    один такой ряд ломает KMeans-кластеризацию (см. clustering.py)."""

    def test_all_zero_metrics_not_stored(self, db_session):
        stored = ingest(
            db_session,
            game="dota2",
            collector=_OnePlayerCollector(),
            normalize_player=lambda raw: raw,
            normalize_snapshot=lambda raw: {"metrics": {"winrate": 0.0, "kda": 0.0, "gpm": 0.0}, "role": None},
        )
        assert stored == 0
        assert db_session.query(Player).count() == 0

    def test_partial_zero_metrics_still_stored(self, db_session):
        # только полностью нулевой набор - сигнал "нет данных"; частично
        # нулевые метрики (например, реальный 0% хедшотов) - обычные данные
        stored = ingest(
            db_session,
            game="dota2",
            collector=_OnePlayerCollector(),
            normalize_player=lambda raw: raw,
            normalize_snapshot=lambda raw: {"metrics": {"winrate": 0.5, "kda": 0.0, "gpm": 0.0}, "role": None},
        )
        assert stored == 1


# lane_role/gold_per_min, при которых normalize._determine_role() выдаёт нужную
# роль - см. tests/test_normalize.py::TestNormalizeOpenDotaSnapshot для той же
# логики по отдельности
_ROLE_MATCH = {
    "1": {"lane_role": 1, "gold_per_min": 500},  # safe, core
    "2": {"lane_role": 2, "gold_per_min": 500},  # mid
    "3": {"lane_role": 3, "gold_per_min": 500},  # off, core
    "4": {"lane_role": 3, "gold_per_min": 100},  # off, support
    "5": {"lane_role": 1, "gold_per_min": 100},  # safe, support
}


class _RoleAwareOpenDotaCollector:
    """Фейковый OpenDota-коллектор без сети: каждому account_id заранее
    назначена желаемая роль через role_by_id (dict сохраняет порядок вставки -
    это и есть порядок пула, как будто отдаёт /proPlayers)."""

    game = "dota2"

    def __init__(self, role_by_id: dict[str, str]):
        self._role_by_id = role_by_id
        self.fetched_ids: list[str] = []

    def fetch_player_pool(self):
        return [{"account_id": aid, "name": f"p{aid}", "team_name": None} for aid in self._role_by_id]

    def fetch_player_stats(self, external_id):
        self.fetched_ids.append(external_id)
        match = _ROLE_MATCH[self._role_by_id[external_id]]
        return {
            "wl": {"win": 5, "lose": 5},
            "totals": [
                {"field": "kills", "n": 1, "sum": 5},
                {"field": "deaths", "n": 1, "sum": 5},
                {"field": "assists", "n": 1, "sum": 5},
                {"field": "gold_per_min", "n": 1, "sum": 500},
                {"field": "xp_per_min", "n": 1, "sum": 500},
                {"field": "hero_damage", "n": 1, "sum": 10000},
            ],
            "matches": [match] * 3,
        }


class TestIngestDota2RoleBackfill:
    """Регресс: optimize_team() гарантированно Infeasible, если у роли 0
    кандидатов (см. ToDoList.md, 2026-08-10) - ingest_dota2() должен добирать
    недостающие роли за пределами обычного среза пула (ТЗ 5.5)."""

    def test_backfills_missing_role_from_extended_pool(self, db_session, monkeypatch):
        monkeypatch.setattr(scheduler, "MIN_CANDIDATES_PER_ROLE", 1)
        role_by_id = {
            "1": "1",
            "2": "3",
            "3": "4",
            "4": "5",  # первые 4 (primary) - роли "2" среди них нет
            "5": "2",  # extended - закрывает недостающую роль "2"
            "6": "1",  # extended - уже не нужен, не должен опрашиваться
        }
        collector = _RoleAwareOpenDotaCollector(role_by_id)

        stored = ingest_dota2(db_session, collector=collector, pool_limit=4)

        assert stored == 5
        assert collector.fetched_ids == ["1", "2", "3", "4", "5"]  # "6" не тронут

        roles = {p.role for p in db_session.query(Player).filter(Player.game == "dota2").all()}
        assert roles == {"1", "2", "3", "4", "5"}

        backfilled = db_session.query(Player).filter_by(game="dota2", external_id="5").one()
        backfilled_snapshot = db_session.query(PlayerSnapshot).filter_by(player_id=backfilled.id).one()
        assert backfilled_snapshot.metrics[BELOW_POOL_THRESHOLD_KEY] is True

        primary = db_session.query(Player).filter_by(game="dota2", external_id="1").one()
        primary_snapshot = db_session.query(PlayerSnapshot).filter_by(player_id=primary.id).one()
        assert BELOW_POOL_THRESHOLD_KEY not in primary_snapshot.metrics

    def test_no_backfill_when_roles_already_covered(self, db_session, monkeypatch):
        monkeypatch.setattr(scheduler, "MIN_CANDIDATES_PER_ROLE", 1)
        role_by_id = {"1": "2", "2": "3", "3": "4", "4": "5", "5": "1", "6": "1"}
        collector = _RoleAwareOpenDotaCollector(role_by_id)

        stored = ingest_dota2(db_session, collector=collector, pool_limit=5)

        assert stored == 5
        assert "6" not in collector.fetched_ids  # extended не опрашивался - роли уже покрыты

    def test_logs_warning_when_role_stays_missing_after_backfill(self, db_session, monkeypatch, caplog):
        monkeypatch.setattr(scheduler, "MIN_CANDIDATES_PER_ROLE", 1)
        monkeypatch.setattr(scheduler, "DOTA2_ROLE_BACKFILL_LIMIT", 2)
        role_by_id = {
            "1": "1",
            "2": "3",
            "3": "4",
            "4": "5",  # primary - роли "2" нет
            "5": "1",
            "6": "1",  # extended - роли "2" тоже нет
        }
        collector = _RoleAwareOpenDotaCollector(role_by_id)

        with caplog.at_level(logging.WARNING, logger="app.scheduler"):
            stored = ingest_dota2(db_session, collector=collector, pool_limit=4)

        assert stored == 6  # все сохранены, просто роль "2" так и не набралась
        assert any("2" in record.getMessage() for record in caplog.records if "добрать" in record.getMessage())
