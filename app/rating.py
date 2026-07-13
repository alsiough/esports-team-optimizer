from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Player, PlayerSnapshot, Rating

RATING_VERSION = "v1"

# Простая объяснимая модель: рейтинг = взвешенная сумма z-score показателей
# относительно других игроков той же игры (и роли - для Dota 2, где роль уже
# известна из normalize.py; для CS2 роль/кластер пока не посчитаны - группа
# сравнения - вся игра, см. TECHNICAL_SPEC.md 5.4). Веса равные по умолчанию,
# константа ниже - точка настройки без изменения логики.
DOTA2_WEIGHTS = {"winrate": 1.0, "kda": 1.0, "gpm": 1.0, "xpm": 1.0, "hero_damage": 1.0}
CS2_WEIGHTS = {"winrate": 1.0, "kd": 1.0, "adr": 1.0, "hs_pct": 1.0}

GAME_WEIGHTS = {"dota2": DOTA2_WEIGHTS, "cs2": CS2_WEIGHTS}
GAME_GROUP_FIELD = {"dota2": "role", "cs2": None}


def _latest_snapshots(session: Session, game: str) -> list[tuple[Player, PlayerSnapshot]]:
    latest_per_player = (
        session.query(
            PlayerSnapshot.player_id,
            func.max(PlayerSnapshot.taken_at).label("latest_taken_at"),
        )
        .join(Player)
        .filter(Player.game == game)
        .group_by(PlayerSnapshot.player_id)
        .subquery()
    )
    return (
        session.query(Player, PlayerSnapshot)
        .join(PlayerSnapshot, PlayerSnapshot.player_id == Player.id)
        .join(
            latest_per_player,
            (latest_per_player.c.player_id == PlayerSnapshot.player_id)
            & (latest_per_player.c.latest_taken_at == PlayerSnapshot.taken_at),
        )
        .filter(Player.game == game)
        .all()
    )


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _zscore_by_group(df: pd.DataFrame, metric_cols: list[str], group_col: str | None) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    if group_col is None:
        for col in metric_cols:
            z[col] = _zscore(df[col])
    else:
        groups = df[group_col].fillna("__unknown__")
        for col in metric_cols:
            z[col] = df.groupby(groups)[col].transform(_zscore)
    return z


def compute_ratings(session: Session, game: str, rating_version: str = RATING_VERSION) -> int:
    """Пересчитывает рейтинг всех игроков игры по последнему снапшоту каждого.

    Вызывается при каждом обновлении снапшотов (см. scheduler.py). Возвращает
    число сохранённых записей рейтинга.
    """
    weights = GAME_WEIGHTS.get(game)
    if weights is None:
        raise ValueError(f"неизвестная игра: {game}")

    rows = _latest_snapshots(session, game)
    if not rows:
        return 0

    df = pd.DataFrame(
        [{"player_id": player.id, "role": player.role, **snapshot.metrics} for player, snapshot in rows]
    )
    metric_cols = list(weights.keys())
    for col in metric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    z = _zscore_by_group(df, metric_cols, GAME_GROUP_FIELD.get(game))
    rating_values = sum(z[col] * weight for col, weight in weights.items())

    now = dt.datetime.now(dt.timezone.utc)
    for player_id, value in zip(df["player_id"], rating_values):
        session.add(
            Rating(
                player_id=int(player_id),
                rating_value=float(value),
                rating_version=rating_version,
                computed_at=now,
            )
        )

    session.commit()
    return len(df)
