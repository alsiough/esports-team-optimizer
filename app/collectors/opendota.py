from __future__ import annotations

from typing import Any

import httpx

from app.collectors.base import BaseCollector, SourceUnavailableError

BASE_URL = "https://api.opendota.com/api"

# поля матча, которые запрашиваем сверх набора по умолчанию (project=... повторяется в query)
MATCH_PROJECT_FIELDS = [
    "kills",
    "deaths",
    "assists",
    "gold_per_min",
    "xp_per_min",
    "hero_damage",
    "lane_role",
    "duration",
    "radiant_win",
    "player_slot",
]
MATCHES_LIMIT = 100


class OpenDotaCollector(BaseCollector):
    game = "dota2"
    min_interval_seconds = 1.1  # ~54 запроса/мин, с запасом от лимита 60/мин

    def __init__(self, api_key: str | None = None, timeout: float = 15.0) -> None:
        super().__init__()
        self._api_key = api_key
        self._timeout = timeout

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(extra or {})
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def fetch_player_pool(self) -> list[dict[str, Any]]:
        with httpx.Client(base_url=BASE_URL, timeout=self._timeout) as client:
            data = self._request_json(client, "GET", "/proPlayers", params=self._params())
        if not isinstance(data, list):
            raise SourceUnavailableError("dota2: /proPlayers вернул неожиданный формат ответа")
        return data

    def fetch_player_stats(self, external_id: str) -> dict[str, Any]:
        with httpx.Client(base_url=BASE_URL, timeout=self._timeout) as client:
            wl = self._request_json(client, "GET", f"/players/{external_id}/wl", params=self._params())
            totals = self._request_json(client, "GET", f"/players/{external_id}/totals", params=self._params())
            matches = self._request_json(
                client,
                "GET",
                f"/players/{external_id}/matches",
                params=self._params({"limit": MATCHES_LIMIT, "project": MATCH_PROJECT_FIELDS}),
            )
        return {"account_id": external_id, "wl": wl, "totals": totals, "matches": matches}
