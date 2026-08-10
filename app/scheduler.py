from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session

from app.clustering import compute_clusters
from app.collectors.base import BaseCollector, SourceUnavailableError
from app.collectors.faceit import KNOWN_REGIONS as CS2_KNOWN_REGIONS
from app.collectors.faceit import FaceitCollector
from app.collectors.opendota import OpenDotaCollector
from app.db import get_session
from app.models import Player, PlayerSnapshot
from app.normalize import (
    normalize_faceit_player,
    normalize_faceit_snapshot,
    normalize_opendota_player,
    normalize_opendota_snapshot,
)
from app.rating import compute_ratings

logger = logging.getLogger(__name__)

NormalizePlayerFn = Callable[[dict[str, Any]], dict[str, Any] | None]
NormalizeSnapshotFn = Callable[[dict[str, Any]], dict[str, Any]]

# Ключ в PlayerSnapshot.metrics, которым помечаются кандидаты, добавленные в
# пул донабором роли (см. ingest_dota2/_backfill_missing_roles) сверх
# обычного среза DOTA2_POOL_LIMIT - индикатор "этот игрок ниже обычной
# границы отбора пула, взят принудительно, чтобы закрыть недостающую роль".
BELOW_POOL_THRESHOLD_KEY = "below_pool_threshold"


def _get_or_create_player(
    session: Session, *, game: str, external_id: str, nickname: str, team: str | None
) -> Player:
    player = session.query(Player).filter_by(game=game, external_id=external_id).one_or_none()
    now = dt.datetime.now(dt.timezone.utc)
    if player is None:
        player = Player(game=game, external_id=external_id, nickname=nickname, team=team)
        session.add(player)
    else:
        player.nickname = nickname
        player.team = team
        player.last_seen = now
    return player


@dataclass
class _IngestOutcome:
    stored: bool
    role: str | None


def _ingest_entry(
    session: Session,
    *,
    game: str,
    collector: BaseCollector,
    normalize_player: NormalizePlayerFn,
    normalize_snapshot: NormalizeSnapshotFn,
    raw_entry: dict[str, Any],
    extra_metrics: dict[str, Any] | None = None,
) -> _IngestOutcome:
    """Обрабатывает одного кандидата из пула: тянет статистику, нормализует,
    пишет снапшот (если данные не деградировавшие). extra_metrics подмешивается
    в metrics снапшота как есть - используется для пометки BELOW_POOL_THRESHOLD_KEY
    при донаборе роли (см. ingest_dota2)."""
    identity = normalize_player(raw_entry)
    if identity is None:
        return _IngestOutcome(stored=False, role=None)

    try:
        raw_stats = collector.fetch_player_stats(identity["external_id"])
    except (SourceUnavailableError, httpx.HTTPError) as exc:
        logger.warning("%s: пропуск игрока %s: %s", game, identity["external_id"], exc)
        return _IngestOutcome(stored=False, role=None)

    snapshot = normalize_snapshot(raw_stats)
    if all(value == 0 for value in snapshot["metrics"].values()):
        # normalize.py осознанно возвращает 0.0 на каждый показатель,
        # когда у источника нет данных по игроку (нет матчей/totals у
        # OpenDota, пустой lifetime у FACEIT) - это "нет сигнала", а не
        # реальный игровой стиль. Такой снапшот не пишем: один подобный
        # ряд полностью ломает KMeans-кластеризацию (см. ToDoList.md,
        # запись 2026-08-10) - он на порядки дальше от остальных точек
        # в масштабированном пространстве признаков, чем они друг от
        # друга, и силуэт находит "оптимальным" выделить его в отдельный
        # кластер, свалив всех остальных игроков в один.
        logger.warning(
            "%s: пропуск игрока %s - все метрики нулевые (нет данных источника)",
            game,
            identity["external_id"],
        )
        return _IngestOutcome(stored=False, role=None)

    player = _get_or_create_player(
        session,
        game=game,
        external_id=identity["external_id"],
        nickname=identity["nickname"],
        team=identity["team"],
    )
    if snapshot["role"] is not None:
        player.role = str(snapshot["role"])
    session.flush()

    metrics = {**snapshot["metrics"], **(extra_metrics or {})}
    session.add(
        PlayerSnapshot(
            player_id=player.id,
            taken_at=dt.datetime.now(dt.timezone.utc),
            metrics=metrics,
        )
    )
    # Коммит на каждого игрока, а не один раз в конце цикла: иначе
    # write-транзакция остаётся открытой на всё время опроса пула
    # (десятки сетевых запросов к внешнему API), и параллельный job
    # другой игры (см. scheduler.create_scheduler) гарантированно
    # ловит "database is locked" на SQLite, а не просто ждёт.
    session.commit()
    return _IngestOutcome(stored=True, role=player.role)


def ingest(
    session: Session,
    *,
    game: str,
    collector: BaseCollector,
    normalize_player: NormalizePlayerFn,
    normalize_snapshot: NormalizeSnapshotFn,
    pool_limit: int | None = None,
) -> int:
    """Опрашивает источник и пишет по одному новому снапшоту на игрока. Возвращает число сохранённых снапшотов."""
    try:
        pool = collector.fetch_player_pool()
    except SourceUnavailableError as exc:
        logger.error("%s: пул игроков недоступен: %s", game, exc)
        return 0

    if pool_limit is not None:
        pool = pool[:pool_limit]

    stored = 0
    for raw_entry in pool:
        outcome = _ingest_entry(
            session,
            game=game,
            collector=collector,
            normalize_player=normalize_player,
            normalize_snapshot=normalize_snapshot,
            raw_entry=raw_entry,
        )
        if outcome.stored:
            stored += 1

    return stored


DOTA2_ROLES = ("1", "2", "3", "4", "5")
# По ТЗ 5.5 optimize_team() требует ровно одного кандидата на каждую роль
# 1-5 - при нуле кандидатов на роль результат гарантированно Infeasible
# независимо от team_size/active_days (см. ToDoList.md, запись 2026-08-10).
# Требуем не 1, а MIN_CANDIDATES_PER_ROLE - запас на случай, если часть
# кандидатов позже отсеется фильтром active_days в оптимизаторе.
MIN_CANDIDATES_PER_ROLE = int(os.getenv("DOTA2_MIN_CANDIDATES_PER_ROLE", "3"))
# Сколько дополнительных кандидатов сверх DOTA2_POOL_LIMIT можно опросить в
# попытке добрать недостающие роли - жёсткая граница бюджета OpenDota,
# иначе при неудачном донаборе можно уйти в сканирование всего /proPlayers.
DOTA2_ROLE_BACKFILL_LIMIT = int(os.getenv("DOTA2_ROLE_BACKFILL_LIMIT", "60"))


def _backfill_missing_roles(
    session: Session,
    collector: OpenDotaCollector,
    candidates: list[dict[str, Any]],
    role_counts: dict[str, int],
) -> int:
    """Донабирает кандидатов за пределами обычного среза пула, пока не
    наберётся MIN_CANDIDATES_PER_ROLE на каждую роль (или не кончится
    candidates/DOTA2_ROLE_BACKFILL_LIMIT). Кандидаты, добавленные так,
    помечаются BELOW_POOL_THRESHOLD_KEY - они не попали в обычный срез пула
    по позиции в /proPlayers, но нужны для гарантии покрытия ролей."""
    missing = {r for r in DOTA2_ROLES if role_counts[r] < MIN_CANDIDATES_PER_ROLE}
    stored = 0
    for raw_entry in candidates[:DOTA2_ROLE_BACKFILL_LIMIT]:
        if not missing:
            break
        outcome = _ingest_entry(
            session,
            game="dota2",
            collector=collector,
            normalize_player=normalize_opendota_player,
            normalize_snapshot=normalize_opendota_snapshot,
            raw_entry=raw_entry,
            extra_metrics={BELOW_POOL_THRESHOLD_KEY: True},
        )
        if outcome.stored:
            stored += 1
            if outcome.role in role_counts:
                role_counts[outcome.role] += 1
                if role_counts[outcome.role] >= MIN_CANDIDATES_PER_ROLE:
                    missing.discard(outcome.role)

    if missing:
        logger.warning(
            "dota2: не удалось добрать минимум %s кандидатов на роли %s даже расширенным поиском (проверено до %s доп. кандидатов)",
            MIN_CANDIDATES_PER_ROLE,
            sorted(missing),
            DOTA2_ROLE_BACKFILL_LIMIT,
        )
    return stored


def ingest_dota2(session: Session, collector: OpenDotaCollector | None = None, pool_limit: int | None = None) -> int:
    # OPENDOTA_API_KEY опционален (в отличие от FACEIT_API_KEY) - публичный
    # лимит и без ключа работает, ключ только поднимает лимиты (см. CLAUDE.md)
    collector = collector or OpenDotaCollector(api_key=os.getenv("OPENDOTA_API_KEY"))
    limit = DOTA2_POOL_LIMIT if pool_limit is None else pool_limit

    try:
        pool = collector.fetch_player_pool()
    except SourceUnavailableError as exc:
        logger.error("dota2: пул игроков недоступен: %s", exc)
        return 0

    primary, extended = pool[:limit], pool[limit:]

    stored = 0
    role_counts: dict[str, int] = {r: 0 for r in DOTA2_ROLES}
    for raw_entry in primary:
        outcome = _ingest_entry(
            session,
            game="dota2",
            collector=collector,
            normalize_player=normalize_opendota_player,
            normalize_snapshot=normalize_opendota_snapshot,
            raw_entry=raw_entry,
        )
        if outcome.stored:
            stored += 1
            if outcome.role in role_counts:
                role_counts[outcome.role] += 1

    if any(role_counts[r] < MIN_CANDIDATES_PER_ROLE for r in DOTA2_ROLES) and extended:
        stored += _backfill_missing_roles(session, collector, extended, role_counts)

    return stored


def ingest_cs2(session: Session, collector: FaceitCollector, pool_limit: int | None = None) -> int:
    return ingest(
        session,
        game="cs2",
        collector=collector,
        normalize_player=normalize_faceit_player,
        normalize_snapshot=normalize_faceit_snapshot,
        pool_limit=pool_limit,
    )


# Бюджет OpenDota - 60 запросов/мин и 50 000/мес; при пуле в тысячи про-игроков
# опрашивать его целиком каждый цикл нельзя (см. TECHNICAL_SPEC.md 11), поэтому
# планировщик берёт ограниченный срез пула за один запуск. Интервалы и размеры
# срезов - конфигурируемые, чтобы их можно было подстроить без правки кода.
# DOTA2_POOL_LIMIT=60 при интервале 60 минут и 3 запросах на игрока (wl/totals/matches)
# даёт ~181 запрос/опрос - с запасом укладывается в лимит 60/мин (троттлинг
# 1.1с/запрос в OpenDotaCollector растягивает опрос на ~3.5 минуты); для
# непрерывной работы 24/7 это уже заметная доля месячного бюджета, но
# локальный демо-проект так не работает - см. ToDoList.md.
# Поднято с 30: /proPlayers отдаёт пул в порядке возрастания account_id, и в
# первых 30 записях иногда не попадается ни одного игрока с ролью "2" (мид) -
# optimize_team() тогда гарантированно Infeasible, т.к. по ТЗ 5.5 нужен ровно
# один игрок на каждую роль 1-5. При пуле 60 роль "2" на практике встречается
# (проверено живым переопросом, см. ToDoList.md запись 2026-08-10).
DOTA2_POLL_INTERVAL_MINUTES = int(os.getenv("DOTA2_POLL_INTERVAL_MINUTES", "60"))
DOTA2_POOL_LIMIT = int(os.getenv("DOTA2_POOL_LIMIT", "60"))
CS2_POLL_INTERVAL_MINUTES = int(os.getenv("CS2_POLL_INTERVAL_MINUTES", "60"))
CS2_POOL_LIMIT = int(os.getenv("CS2_POOL_LIMIT", "100"))
# Список регионов через запятую - FaceitCollector объединяет их топ-рейтинги
# вперемешку (round-robin), см. app/collectors/faceit.py:KNOWN_REGIONS.
CS2_REGIONS = [
    r.strip() for r in os.getenv("FACEIT_REGIONS", ",".join(CS2_KNOWN_REGIONS)).split(",") if r.strip()
]


@dataclass
class RefreshOutcome:
    snapshots_stored: int
    players_rated: int
    players_clustered: int


def refresh_game(
    session: Session,
    game: str,
    *,
    dota2_pool_limit: int = DOTA2_POOL_LIMIT,
    cs2_pool_limit: int = CS2_POOL_LIMIT,
    cs2_regions: list[str] = CS2_REGIONS,
) -> RefreshOutcome:
    """Внеочередной опрос источника game + пересчёт рейтинга и кластеров.

    Общая точка входа для планового job'а, POST /refresh (app/api.py) и
    кнопки "Опросить источник" в дашборде - без неё ingest+compute_ratings+
    compute_clusters дублировались бы в каждом из мест по отдельности.
    """
    if game == "dota2":
        stored = ingest_dota2(session, pool_limit=dota2_pool_limit)
    elif game == "cs2":
        api_key = os.getenv("FACEIT_API_KEY")
        if not api_key:
            raise RuntimeError("cs2: FACEIT_API_KEY не задан")
        stored = ingest_cs2(session, FaceitCollector(api_key=api_key, regions=cs2_regions), pool_limit=cs2_pool_limit)
    else:
        raise ValueError(f"неизвестная игра: {game}")

    rated = compute_ratings(session, game) if stored else 0
    clustered = compute_clusters(session, game) if stored else 0
    return RefreshOutcome(snapshots_stored=stored, players_rated=rated, players_clustered=clustered)


def _job_ingest_dota2() -> None:
    # Плановый job намеренно пересчитывает только рейтинг, не кластеры -
    # кластеризация тяжелее и не обязана поспевать за каждым интервалом
    # опроса; см. refresh_game() выше для сценариев (API/дашборд), где нужен
    # полный пересчёт по требованию.
    session = get_session()
    try:
        stored = ingest_dota2(session, pool_limit=DOTA2_POOL_LIMIT)
        logger.info("dota2: job завершён, снапшотов сохранено: %s", stored)
        if stored:
            rated = compute_ratings(session, "dota2")
            logger.info("dota2: рейтинг пересчитан для %s игроков", rated)
    except Exception:
        logger.exception("dota2: job упал с ошибкой")
    finally:
        session.close()


def _job_ingest_cs2() -> None:
    api_key = os.getenv("FACEIT_API_KEY")
    if not api_key:
        logger.warning("cs2: FACEIT_API_KEY не задан, job пропущен")
        return
    session = get_session()
    try:
        collector = FaceitCollector(api_key=api_key, regions=CS2_REGIONS)
        stored = ingest_cs2(session, collector, pool_limit=CS2_POOL_LIMIT)
        logger.info("cs2: job завершён, снапшотов сохранено: %s", stored)
        if stored:
            rated = compute_ratings(session, "cs2")
            logger.info("cs2: рейтинг пересчитан для %s игроков", rated)
    except Exception:
        logger.exception("cs2: job упал с ошибкой")
    finally:
        session.close()


# Кластеризация не пересчитывается на каждом ingest-job'е (см. комментарии
# внутри _job_ingest_dota2/_job_ingest_cs2) - она заметно тяжелее одного
# опроса (KMeans перебирает k=2..8 с n_init=10 на каждый) и не обязана
# поспевать за каждым коротким интервалом. Но без ПЕРИОДИЧЕСКОГО пересчёта
# кластеры бесконечно отстают от новых игроков - на практике застряли на
# 27 дней, и 66 из 197 cs2-игроков вообще ни разу не кластеризовались (см.
# ToDoList.md, 2026-08-10). Поэтому - отдельный job с более длинным
# интервалом, а не на каждом цикле ingest.
CLUSTER_RECOMPUTE_INTERVAL_MINUTES = int(os.getenv("CLUSTER_RECOMPUTE_INTERVAL_MINUTES", "360"))


def _job_recompute_clusters() -> None:
    session = get_session()
    try:
        for game in ("dota2", "cs2"):
            clustered = compute_clusters(session, game)
            logger.info("%s: кластеры пересчитаны для %s игроков", game, clustered)
    except Exception:
        logger.exception("recompute_clusters: job упал с ошибкой")
    finally:
        session.close()


def create_scheduler() -> BackgroundScheduler:
    """Собирает BackgroundScheduler с job'ами опроса обоих источников и
    периодического пересчёта кластеров. Не запускает - вызывающий код сам
    решает, когда start()/shutdown()."""
    scheduler = BackgroundScheduler(
        timezone="UTC",
        executors={"default": ThreadPoolExecutor(max_workers=3)},
    )
    now = dt.datetime.now(dt.timezone.utc)
    scheduler.add_job(
        _job_ingest_dota2,
        trigger=IntervalTrigger(minutes=DOTA2_POLL_INTERVAL_MINUTES),
        id="ingest_dota2",
        next_run_time=now,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _job_ingest_cs2,
        trigger=IntervalTrigger(minutes=CS2_POLL_INTERVAL_MINUTES),
        id="ingest_cs2",
        next_run_time=now,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _job_recompute_clusters,
        trigger=IntervalTrigger(minutes=CLUSTER_RECOMPUTE_INTERVAL_MINUTES),
        id="recompute_clusters",
        next_run_time=now,
        max_instances=1,
        coalesce=True,
    )
    return scheduler
