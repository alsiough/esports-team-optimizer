from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (UniqueConstraint("game", "external_id", name="uq_players_game_external_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game: Mapped[str] = mapped_column(String(16), nullable=False)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)
    nickname: Mapped[str] = mapped_column(String(128), nullable=False)
    team: Mapped[str | None] = mapped_column(String(128), nullable=True)
    role: Mapped[str | None] = mapped_column(String(8), nullable=True)
    first_seen: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_seen: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    snapshots: Mapped[list["PlayerSnapshot"]] = relationship(back_populates="player", cascade="all, delete-orphan")
    clusters: Mapped[list["Cluster"]] = relationship(back_populates="player", cascade="all, delete-orphan")
    ratings: Mapped[list["Rating"]] = relationship(back_populates="player", cascade="all, delete-orphan")


class PlayerSnapshot(Base):
    __tablename__ = "player_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    taken_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)

    player: Mapped["Player"] = relationship(back_populates="snapshots")


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    cluster_label: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    player: Mapped["Player"] = relationship(back_populates="clusters")


class Rating(Base):
    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    rating_value: Mapped[float] = mapped_column(Float, nullable=False)
    rating_version: Mapped[str] = mapped_column(String(32), nullable=False)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    player: Mapped["Player"] = relationship(back_populates="ratings")


class TeamProposal(Base):
    __tablename__ = "team_proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    constraints: Mapped[dict] = mapped_column(JSON, nullable=False)
    player_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    objective_value: Mapped[float] = mapped_column(Float, nullable=False)
