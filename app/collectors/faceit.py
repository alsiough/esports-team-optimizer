from __future__ import annotations

from typing import Any

import httpx

from app.collectors.base import BaseCollector, SourceUnavailableError

BASE_URL = "https://open.faceit.com/data/v4"
CS2_GAME_ID = "cs2"
DEFAULT_REGION = "EU"
POOL_LIMIT = 100


class FaceitCollector(BaseCollector):
    game = "cs2"
    min_interval_seconds = 0.6  # без документированного публичного лимита - держим запас

    def __init__(
        self,
        api_key: str,
        region: str = DEFAULT_REGION,
        game_id: str = CS2_GAME_ID,
        timeout: float = 15.0,
    ) -> None:
        super().__init__()
        if not api_key:
            raise ValueError("FACEIT_API_KEY не задан")
        self._api_key = api_key
        self._region = region
        self._game_id = game_id
        self._timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=BASE_URL,
            timeout=self._timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    def fetch_player_pool(self) -> list[dict[str, Any]]:
        with self._client() as client:
            data = self._request_json(
                client,
                "GET",
                f"/rankings/games/{self._game_id}/regions/{self._region}",
                params={"limit": POOL_LIMIT},
            )
        items = data.get("items") if isinstance(data, dict) else None
        if items is None:
            raise SourceUnavailableError(
                f"cs2: /rankings/games/{self._game_id}/regions/{self._region} вернул неожиданный формат ответа"
            )
        return items

    def fetch_player_stats(self, external_id: str) -> dict[str, Any]:
        with self._client() as client:
            stats = self._request_json(client, "GET", f"/players/{external_id}/stats/{self._game_id}")
        return {"player_id": external_id, "stats": stats}
