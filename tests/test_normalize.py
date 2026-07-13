"""Тесты normalize.py - сырые ответы источников -> единая схема снапшота (ТЗ 5.2, 6.2)."""

from __future__ import annotations

from app.normalize import (
    normalize_faceit_player,
    normalize_faceit_snapshot,
    normalize_opendota_player,
    normalize_opendota_snapshot,
)


def _match(lane_role: int | None = None, gpm: float = 0) -> dict:
    return {"lane_role": lane_role, "gold_per_min": gpm}


class TestNormalizeOpenDotaPlayer:
    def test_valid(self):
        raw = {"account_id": 123, "name": "Nickname", "team_name": "Team A"}
        assert normalize_opendota_player(raw) == {
            "external_id": "123",
            "nickname": "Nickname",
            "team": "Team A",
        }

    def test_falls_back_to_personaname(self):
        result = normalize_opendota_player({"account_id": 123, "personaname": "Persona"})
        assert result["nickname"] == "Persona"
        assert result["team"] is None

    def test_missing_account_id_returns_none(self):
        assert normalize_opendota_player({"name": "X"}) is None

    def test_missing_nickname_returns_none(self):
        assert normalize_opendota_player({"account_id": 1}) is None


class TestNormalizeOpenDotaSnapshot:
    def test_winrate_and_totals(self):
        raw = {
            "wl": {"win": 7, "lose": 3},
            "totals": [
                {"field": "kills", "n": 10, "sum": 50},
                {"field": "deaths", "n": 10, "sum": 20},
                {"field": "assists", "n": 10, "sum": 80},
                {"field": "gold_per_min", "n": 10, "sum": 5000},
                {"field": "xp_per_min", "n": 10, "sum": 6000},
                {"field": "hero_damage", "n": 10, "sum": 200000},
            ],
            "matches": [],
        }
        result = normalize_opendota_snapshot(raw)
        assert result["metrics"]["winrate"] == 0.7
        assert result["metrics"]["kda"] == 6.5  # (avg_kills=5 + avg_assists=8) / avg_deaths=2
        assert result["metrics"]["gpm"] == 500
        assert result["metrics"]["xpm"] == 600
        assert result["metrics"]["hero_damage"] == 20000
        assert result["role"] is None  # <3 матчей с непустым lane_role

    def test_winrate_no_games_defaults_to_zero(self):
        result = normalize_opendota_snapshot({"wl": {"win": 0, "lose": 0}, "totals": [], "matches": []})
        assert result["metrics"]["winrate"] == 0.0

    def test_kda_zero_deaths_uses_floor_of_one(self):
        raw = {
            "wl": {},
            "matches": [],
            "totals": [
                {"field": "kills", "n": 1, "sum": 5},
                {"field": "deaths", "n": 1, "sum": 0},
                {"field": "assists", "n": 1, "sum": 3},
            ],
        }
        assert normalize_opendota_snapshot(raw)["metrics"]["kda"] == 8.0

    def test_role_needs_at_least_three_matches(self):
        matches = [_match(lane_role=1, gpm=500), _match(lane_role=1, gpm=500)]
        result = normalize_opendota_snapshot({"wl": {}, "totals": [], "matches": matches})
        assert result["role"] is None

    def test_role_mid_regardless_of_gpm(self):
        matches = [_match(lane_role=2, gpm=100)] * 3
        result = normalize_opendota_snapshot({"wl": {}, "totals": [], "matches": matches})
        assert result["role"] == 2

    def test_role_safe_lane_high_gpm_is_core(self):
        matches = [_match(lane_role=1, gpm=500)] * 3
        result = normalize_opendota_snapshot({"wl": {}, "totals": [], "matches": matches})
        assert result["role"] == 1

    def test_role_safe_lane_low_gpm_is_support(self):
        matches = [_match(lane_role=1, gpm=200)] * 3
        result = normalize_opendota_snapshot({"wl": {}, "totals": [], "matches": matches})
        assert result["role"] == 5

    def test_role_off_lane_high_gpm_is_core(self):
        matches = [_match(lane_role=3, gpm=500)] * 3
        result = normalize_opendota_snapshot({"wl": {}, "totals": [], "matches": matches})
        assert result["role"] == 3

    def test_role_off_lane_low_gpm_is_support(self):
        matches = [_match(lane_role=3, gpm=100)] * 3
        result = normalize_opendota_snapshot({"wl": {}, "totals": [], "matches": matches})
        assert result["role"] == 4

    def test_role_by_majority_vote(self):
        matches = [_match(lane_role=1, gpm=500)] * 2 + [_match(lane_role=3, gpm=100)]
        result = normalize_opendota_snapshot({"wl": {}, "totals": [], "matches": matches})
        assert result["role"] == 1  # большинство голосов - safe-лейн


class TestNormalizeFaceitPlayer:
    def test_valid(self):
        raw = {"player_id": "abc-123", "nickname": "s1mple"}
        assert normalize_faceit_player(raw) == {"external_id": "abc-123", "nickname": "s1mple", "team": None}

    def test_missing_player_id_returns_none(self):
        assert normalize_faceit_player({"nickname": "x"}) is None

    def test_missing_nickname_returns_none(self):
        assert normalize_faceit_player({"player_id": "1"}) is None


class TestNormalizeFaceitSnapshot:
    def test_uses_average_fields_not_broken_aggregates(self):
        raw = {
            "stats": {
                "lifetime": {
                    "Win Rate %": "55",
                    "K/D Ratio": "10331.88",  # битый агрегат - не должен использоваться
                    "Average K/D Ratio": "1.15",
                    "Total Headshots %": "99999",  # битый агрегат - не должен использоваться
                    "Average Headshots %": "42",
                    "ADR": "78.5",
                }
            }
        }
        result = normalize_faceit_snapshot(raw)
        assert result["metrics"] == {"winrate": 0.55, "kd": 1.15, "adr": 78.5, "hs_pct": 0.42}
        assert result["role"] is None

    def test_missing_lifetime_defaults_to_zero(self):
        result = normalize_faceit_snapshot({"stats": {}})
        assert result["metrics"] == {"winrate": 0.0, "kd": 0.0, "adr": 0.0, "hs_pct": 0.0}

    def test_non_numeric_value_falls_back_to_default(self):
        raw = {"stats": {"lifetime": {"Average K/D Ratio": "not-a-number"}}}
        assert normalize_faceit_snapshot(raw)["metrics"]["kd"] == 0.0
