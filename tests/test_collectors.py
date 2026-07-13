"""Тесты collectors/* - HTTP-адаптеры на моках (httpx.MockTransport), без реальной сети.

Покрывает: успешные ответы, неожиданный формат ответа, backoff/retry на
429/5xx (см. app/collectors/base.py), исчерпание попыток и граничный случай
"пустой пул" (ТЗ 10 - критерии приёмки/тестирование).
"""

from __future__ import annotations

import httpx
import pytest

from app.collectors.base import SourceUnavailableError
from app.collectors.faceit import FaceitCollector
from app.collectors.opendota import OpenDotaCollector
from app.scheduler import ingest


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
