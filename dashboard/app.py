"""Streamlit-дашборд - см. TECHNICAL_SPEC.md 5.6.

Читает и пишет БД напрямую через модули app.* (rating/clustering/optimizer/
scheduler) - работает независимо от того, запущен ли `uvicorn app.api:app`
(см. CLAUDE.md: backend и дашборд - отдельные команды).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run dashboard/app.py` из корня проекта кладёт в sys.path каталог
# dashboard/, а не корень - без этого `import app.*` не находится.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from app.clustering import latest_clusters, pca_2d
from app.db import get_session, init_db
from app.models import Player, PlayerSnapshot, Rating
from app.optimizer import (
    DEFAULT_ACTIVE_DAYS,
    DEFAULT_MAX_PER_CLUSTER,
    DEFAULT_MIN_CLUSTERS,
    TEAM_SIZE,
    optimize_team,
)
from app.rating import latest_ratings
from app.scheduler import refresh_game

GAMES = ("dota2", "cs2")
GAME_LABELS = {"dota2": "Dota 2", "cs2": "CS2"}

st.set_page_config(page_title="Esports Team Optimizer", layout="wide")
init_db()
session = get_session()

st.title("Подбор состава киберспортивной команды")

game = st.sidebar.selectbox("Игра", GAMES, format_func=lambda g: GAME_LABELS[g])

last_taken_at = (
    session.query(PlayerSnapshot.taken_at)
    .join(Player)
    .filter(Player.game == game)
    .order_by(PlayerSnapshot.taken_at.desc())
    .limit(1)
    .scalar()
)
if last_taken_at:
    st.sidebar.caption(f"Последний опрос источника: {last_taken_at:%Y-%m-%d %H:%M} UTC")
else:
    st.sidebar.caption("Данных ещё нет — нажмите «Опросить источник сейчас»")

if st.sidebar.button("Опросить источник сейчас"):
    with st.spinner(f"Опрашиваем источник ({GAME_LABELS[game]})..."):
        try:
            outcome = refresh_game(session, game)
            st.sidebar.success(
                f"Готово: снапшотов {outcome.snapshots_stored}, "
                f"рейтингов {outcome.players_rated}, кластеров {outcome.players_clustered}"
            )
        except RuntimeError as exc:
            st.sidebar.error(str(exc))
    st.rerun()

ratings = latest_ratings(session, game)
clusters = latest_clusters(session, game)
players = session.query(Player).filter(Player.game == game).order_by(Player.nickname).all()

players_df = pd.DataFrame(
    [
        {
            "id": p.id,
            "nickname": p.nickname,
            "team": p.team,
            "role": p.role,
            "cluster": clusters.get(p.id),
            "rating": ratings.get(p.id),
            "last_seen": p.last_seen,
        }
        for p in players
    ]
)

st.header("Игроки")
if players_df.empty:
    st.info("Игроков для этой игры пока нет — опросите источник.")
else:
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        cluster_options = sorted(int(c) for c in players_df["cluster"].dropna().unique())
        cluster_filter = st.selectbox("Кластер", ["Все"] + cluster_options)
    known_ratings = players_df["rating"].dropna()
    rating_bounds = (float(known_ratings.min()), float(known_ratings.max())) if not known_ratings.empty else (-5.0, 5.0)
    with filter_col2:
        min_rating = st.number_input("Мин. рейтинг", value=rating_bounds[0], step=0.5)
    with filter_col3:
        max_rating = st.number_input("Макс. рейтинг", value=rating_bounds[1], step=0.5)

    filtered = players_df
    if cluster_filter != "Все":
        filtered = filtered[filtered["cluster"] == cluster_filter]
    filtered = filtered[filtered["rating"].isna() | filtered["rating"].between(min_rating, max_rating)]
    st.dataframe(filtered, width="stretch", hide_index=True)

st.header("Кластеры (PCA, 2D)")
pca_df = pca_2d(session, game)
if pca_df.empty or "pc1" not in pca_df.columns:
    st.info("Недостаточно данных для PCA-проекции (нужно минимум 2 игрока со снапшотами).")
else:
    pca_df = pca_df.merge(players_df[["id", "nickname"]], left_on="player_id", right_on="id", how="left")
    pca_df["cluster"] = pca_df["player_id"].map(clusters).astype(str)
    chart = (
        alt.Chart(pca_df)
        .mark_circle(size=100)
        .encode(x="pc1", y="pc2", color="cluster:N", tooltip=["nickname", "cluster"])
        .interactive()
    )
    st.altair_chart(chart, width="stretch")

st.header("История рейтинга игрока")
if players_df.empty:
    st.info("Нет игроков для отображения истории.")
else:
    nickname_to_id = dict(zip(players_df["nickname"], players_df["id"]))
    selected_nickname = st.selectbox("Игрок", sorted(nickname_to_id))
    rating_history = (
        session.query(Rating.computed_at, Rating.rating_value)
        .filter(Rating.player_id == nickname_to_id[selected_nickname])
        .order_by(Rating.computed_at)
        .all()
    )
    if not rating_history:
        st.info("У этого игрока пока нет истории рейтинга.")
    else:
        history_df = pd.DataFrame(rating_history, columns=["computed_at", "rating_value"]).set_index("computed_at")
        st.line_chart(history_df)

st.header("Подбор оптимального состава")
with st.form("optimize_form"):
    team_size = st.number_input("Размер состава", min_value=1, value=TEAM_SIZE, step=1)
    active_days = st.number_input("Активность кандидатов, дней", min_value=1, value=DEFAULT_ACTIVE_DAYS, step=1)
    if game == "cs2":
        max_per_cluster = st.number_input("MAX_PER_CLUSTER", min_value=1, value=DEFAULT_MAX_PER_CLUSTER, step=1)
        min_clusters = st.number_input("MIN_CLUSTERS", min_value=1, value=DEFAULT_MIN_CLUSTERS, step=1)
    else:
        max_per_cluster, min_clusters = DEFAULT_MAX_PER_CLUSTER, DEFAULT_MIN_CLUSTERS
        st.caption("Для Dota 2 состав собирается по ролям 1-5, MAX_PER_CLUSTER/MIN_CLUSTERS не применяются.")
    submitted = st.form_submit_button("Подобрать состав")

if submitted:
    result = optimize_team(
        session,
        game,
        team_size=int(team_size),
        max_per_cluster=int(max_per_cluster),
        min_clusters=int(min_clusters),
        active_days=int(active_days),
    )
    if not result.feasible:
        st.error(f"Не удалось собрать состав: {result.message}")
    else:
        st.success(f"Состав найден, суммарный рейтинг: {result.objective_value:.3f}")
        team_df = pd.DataFrame(
            [
                {
                    "nickname": session.get(Player, pid).nickname,
                    "role": session.get(Player, pid).role,
                    "cluster": clusters.get(pid),
                    "rating": ratings.get(pid),
                }
                for pid in result.player_ids
            ]
        )
        st.dataframe(team_df, width="stretch", hide_index=True)

session.close()
