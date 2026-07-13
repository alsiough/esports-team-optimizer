from __future__ import annotations

from itertools import zip_longest
from typing import Any

import httpx

from app.collectors.base import BaseCollector, SourceUnavailableError

BASE_URL = "https://open.faceit.com/data/v4"
CS2_GAME_ID = "cs2"
DEFAULT_REGION = "EU"
# Регионы проверены вживую на реальном API (GET .../rankings/games/cs2/regions/{region}) -
# только эти вернули непустой items на момент проверки; другие правдоподобные коды
# (US, AF, AS, OC, ME, MENA, APAC) существуют как параметр (200 OK), но с пустым списком.
KNOWN_REGIONS = ("EU", "NA", "SA", "OCE", "SEA")
POOL_LIMIT = 100


class FaceitCollector(BaseCollector):
    game = "cs2"
    min_interval_seconds = 0.6  # без документированного публичного лимита - держим запас

    def __init__(
        self,
        api_key: str,
        regions: str | list[str] = DEFAULT_REGION,
        game_id: str = CS2_GAME_ID,
        timeout: float = 15.0,
    ) -> None:
        super().__init__()
        if not api_key:
            raise ValueError("FACEIT_API_KEY не задан")
        self._api_key = api_key
        self._regions = [regions] if isinstance(regions, str) else list(regions)
        self._game_id = game_id
        self._timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=BASE_URL,
            timeout=self._timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    def fetch_player_pool(self) -> list[dict[str, Any]]:
        """Пул кандидатов - объединение топ-N региональных рейтингов (см. KNOWN_REGIONS).

        Списки регионов объединяются вперемешку (round-robin), а не один за другим -
        иначе последующая обрезка пула до pool_limit (см. scheduler.ingest) забрала бы
        только первый регион и разнообразие регионов пропало бы.
        """
        per_region: list[list[dict[str, Any]]] = []
        with self._client() as client:
            for region in self._regions:
                data = self._request_json(
                    client,
                    "GET",
                    f"/rankings/games/{self._game_id}/regions/{region}",
                    params={"limit": POOL_LIMIT},
                )
                items = data.get("items") if isinstance(data, dict) else None
                if items is None:
                    raise SourceUnavailableError(
                        f"cs2: /rankings/games/{self._game_id}/regions/{region} вернул неожиданный формат ответа"
                    )
                per_region.append(items)

        pool: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for row in zip_longest(*per_region):
            for item in row:
                if item is None:
                    continue
                player_id = item.get("player_id")
                if player_id is not None:
                    if player_id in seen_ids:
                        continue
                    seen_ids.add(player_id)
                pool.append(item)
        return pool

    def fetch_player_stats(self, external_id: str) -> dict[str, Any]:
        with self._client() as client:
            stats = self._request_json(client, "GET", f"/players/{external_id}/stats/{self._game_id}")
        return {"player_id": external_id, "stats": stats}
