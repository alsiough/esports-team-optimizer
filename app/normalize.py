from __future__ import annotations

from typing import Any

LANE_SAFE = 1
LANE_MID = 2
LANE_OFF = 3

MIN_MATCHES_FOR_ROLE = 3
# из открытых данных OpenDota нет прямого поля "позиция 1-5", только лейн (safe/mid/off);
# внутри safe/off лейна core и support разделяем по среднему gpm — порог условный,
# используется только как приближение, см. TECHNICAL_SPEC.md 5.2
CORE_GPM_THRESHOLD = 400


def normalize_opendota_player(raw: dict[str, Any]) -> dict[str, Any] | None:
    account_id = raw.get("account_id")
    nickname = raw.get("name") or raw.get("personaname")
    if account_id is None or not nickname:
        return None
    return {
        "external_id": str(account_id),
        "nickname": nickname,
        "team": raw.get("team_name"),
    }


def _totals_avg(totals: list[dict[str, Any]]) -> dict[str, float]:
    by_field = {row["field"]: row for row in totals if "field" in row}

    def avg(field: str) -> float:
        row = by_field.get(field)
        if not row or not row.get("n"):
            return 0.0
        return row["sum"] / row["n"]

    kills, deaths, assists = avg("kills"), avg("deaths"), avg("assists")
    return {
        "kda": (kills + assists) / max(1.0, deaths),
        "gpm": avg("gold_per_min"),
        "xpm": avg("xp_per_min"),
        "hero_damage": avg("hero_damage"),
    }


def _winrate(wl: dict[str, Any]) -> float:
    win, lose = wl.get("win") or 0, wl.get("lose") or 0
    total = win + lose
    return win / total if total else 0.0


def _determine_role(matches: list[dict[str, Any]]) -> int | None:
    votes = [m for m in matches if m.get("lane_role") in (LANE_SAFE, LANE_MID, LANE_OFF)]
    if len(votes) < MIN_MATCHES_FOR_ROLE:
        return None

    counts: dict[int, int] = {}
    for m in votes:
        counts[m["lane_role"]] = counts.get(m["lane_role"], 0) + 1
    dominant_lane = max(counts, key=counts.get)

    if dominant_lane == LANE_MID:
        return 2

    lane_matches = [m for m in votes if m["lane_role"] == dominant_lane]
    avg_gpm = sum(m.get("gold_per_min") or 0 for m in lane_matches) / len(lane_matches)
    is_core = avg_gpm >= CORE_GPM_THRESHOLD

    if dominant_lane == LANE_SAFE:
        return 1 if is_core else 5
    return 3 if is_core else 4


def normalize_opendota_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    metrics = {"winrate": _winrate(raw.get("wl") or {}), **_totals_avg(raw.get("totals") or [])}
    role = _determine_role(raw.get("matches") or [])
    return {"metrics": metrics, "role": role}


def normalize_faceit_player(raw: dict[str, Any]) -> dict[str, Any] | None:
    player_id = raw.get("player_id")
    nickname = raw.get("nickname")
    if not player_id or not nickname:
        return None
    return {
        "external_id": player_id,
        "nickname": nickname,
        "team": None,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_faceit_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    # lifetime отдаёт значения строками; "K/D Ratio" и "Total Headshots %" на этом
    # уровне - битые агрегаты (напр. 10331.88), корректные - "Average ..."-варианты
    lifetime = ((raw.get("stats") or {}).get("lifetime")) or {}
    metrics = {
        "winrate": _safe_float(lifetime.get("Win Rate %")) / 100,
        "kd": _safe_float(lifetime.get("Average K/D Ratio")),
        "adr": _safe_float(lifetime.get("ADR")),
        "hs_pct": _safe_float(lifetime.get("Average Headshots %")) / 100,
    }
    return {"metrics": metrics, "role": None}
