from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.models import Cluster, Player, PlayerSnapshot

DOTA2_FEATURES = ["winrate", "kda", "gpm", "xpm", "hero_damage"]
CS2_FEATURES = ["winrate", "kd", "adr", "hs_pct"]
GAME_FEATURES = {"dota2": DOTA2_FEATURES, "cs2": CS2_FEATURES}

DEFAULT_LOOKBACK_DAYS = 30
K_RANGE = range(2, 9)  # кандидаты числа кластеров, лучший выбирается по силуэту


def _feature_table(session: Session, game: str, lookback_days: int) -> pd.DataFrame:
    """Средние значения метрик по снапшотам за период - по одному ряду на игрока."""
    features = GAME_FEATURES[game]
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=lookback_days)
    rows = (
        session.query(PlayerSnapshot.player_id, PlayerSnapshot.metrics)
        .join(Player)
        .filter(Player.game == game, PlayerSnapshot.taken_at >= since)
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=["player_id", *features])

    df = pd.DataFrame([{"player_id": player_id, **metrics} for player_id, metrics in rows])
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    return df.groupby("player_id", as_index=False)[features].mean()


def _scaled_features(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(df[features].to_numpy())


def _choose_k(x_scaled: np.ndarray, k_range: range) -> int:
    """Число кластеров по силуэту; при нехватке данных - минимально возможное k."""
    best_k, best_score = k_range.start, -1.0
    for k in k_range:
        if k >= len(x_scaled):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(x_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def compute_clusters(
    session: Session,
    game: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    k_range: range = K_RANGE,
) -> int:
    """KMeans-кластеризация по средним метрикам игроков за период. Возвращает число кластеризованных игроков.

    Для Dota 2 результат - уточнение/проверка формальных ролей 1-5 (роль в
    кластеризацию не подаётся, чтобы не задавать ответ заранее). Для CS2
    кластеры используются как псевдо-роли в оптимизаторе, т.к. формальных
    ролей источник не даёт. См. TECHNICAL_SPEC.md 5.3.
    """
    features = GAME_FEATURES.get(game)
    if features is None:
        raise ValueError(f"неизвестная игра: {game}")

    df = _feature_table(session, game, lookback_days)
    if len(df) < 2:
        return 0

    x_scaled = _scaled_features(df, features)
    k = _choose_k(x_scaled, k_range)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x_scaled)

    algorithm = f"kmeans_k{k}"
    now = dt.datetime.now(dt.timezone.utc)
    for player_id, label in zip(df["player_id"], labels):
        session.add(
            Cluster(player_id=int(player_id), algorithm=algorithm, cluster_label=int(label), computed_at=now)
        )
    session.commit()
    return len(df)


def detect_outliers_dbscan(
    session: Session,
    game: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    eps: float = 1.5,
    min_samples: int = 3,
) -> pd.DataFrame:
    """DBSCAN поверх тех же признаков - для сравнения с KMeans и поиска выбросов (label -1). Не пишет в БД."""
    features = GAME_FEATURES.get(game)
    if features is None:
        raise ValueError(f"неизвестная игра: {game}")

    df = _feature_table(session, game, lookback_days)
    if len(df) < min_samples:
        return df.assign(dbscan_label=pd.Series(dtype=int))

    x_scaled = _scaled_features(df, features)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_scaled)
    return df.assign(dbscan_label=labels)


def pca_2d(session: Session, game: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """2D PCA-проекция признаков для визуализации кластеров на дашборде. Не пишет в БД."""
    features = GAME_FEATURES.get(game)
    if features is None:
        raise ValueError(f"неизвестная игра: {game}")

    df = _feature_table(session, game, lookback_days)
    if len(df) < 2:
        return df.assign(pc1=pd.Series(dtype=float), pc2=pd.Series(dtype=float))

    x_scaled = _scaled_features(df, features)
    coords = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
    return df.assign(pc1=coords[:, 0], pc2=coords[:, 1])
