from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SourceUnavailableError(Exception):
    """Источник не ответил после исчерпания всех повторных попыток."""


class BaseCollector(ABC):
    game: str
    min_interval_seconds: float = 0.0

    def __init__(self) -> None:
        self._last_request_at: float = 0.0

    @abstractmethod
    def fetch_player_pool(self) -> list[dict[str, Any]]:
        """Список кандидатов источника (сырые записи, без нормализации)."""

    @abstractmethod
    def fetch_player_stats(self, external_id: str) -> dict[str, Any]:
        """Сырые данные по одному игроку (без нормализации)."""

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait = self.min_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _request_json(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        *,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        attempt = 0
        while True:
            self._throttle()
            try:
                response = client.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                attempt += 1
                if attempt > max_retries:
                    raise SourceUnavailableError(f"{self.game}: {url} недоступен ({exc})") from exc
                self._sleep_backoff(attempt, backoff_base)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                attempt += 1
                if attempt > max_retries:
                    raise SourceUnavailableError(
                        f"{self.game}: {url} вернул {response.status_code} после {max_retries} попыток"
                    )
                retry_after = response.headers.get("Retry-After")
                forced_delay = float(retry_after) if retry_after else None
                logger.warning(
                    "%s: %s -> %s, попытка %s/%s", self.game, url, response.status_code, attempt, max_retries
                )
                self._sleep_backoff(attempt, backoff_base, forced_delay)
                continue

            response.raise_for_status()
            return response.json()

    @staticmethod
    def _sleep_backoff(attempt: int, base: float, forced_delay: float | None = None) -> None:
        delay = forced_delay if forced_delay is not None else base * (2 ** (attempt - 1))
        delay += random.uniform(0, base)
        time.sleep(delay)
