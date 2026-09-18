"""Unit tests for data caching, format filtering, and synthetic dataset generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from chgk_rating.config import DataIngestConfig, FORMAT_OFFLINE, FORMAT_ONLINE, FORMAT_SYNC
from chgk_rating.data_loader import (
    ChGKDataIngester,
    ResponseCache,
    generate_synthetic_dataset,
    infer_participation_format,
    is_aggregate_tournament,
    is_online_venue,
    normalize_tournament_format,
)


def test_response_cache(tmp_path: Path) -> None:
    db_file = tmp_path / "test_cache.db"
    cache = ResponseCache(db_file)

    url = "https://api.rating.chgk.info/tournaments/12345"
    data = {"id": 12345, "name": "Test Tournament"}

    assert cache.get(url) is None
    cache.set(url, 200, data)
    assert cache.get(url) == data

    cache.invalidate(url)
    assert cache.get(url) is None

    # Test sync state persistence
    assert cache.get_sync_state("last_sync") is None
    cache.set_sync_state("last_sync", "2024-05-01T12:00:00")
    assert cache.get_sync_state("last_sync") == "2024-05-01T12:00:00"


def test_tournament_format_normalization() -> None:
    assert normalize_tournament_format("Очный", "Кубок Москвы") == FORMAT_OFFLINE
    assert normalize_tournament_format("Синхрон", "Лига Сибири") == FORMAT_SYNC
    assert normalize_tournament_format("Асинхрон", "Открытый турнир") == FORMAT_SYNC
    assert normalize_tournament_format("Онлайн", "Турнир в Zoom") == FORMAT_ONLINE
    assert normalize_tournament_format("Синхрон", "Онлайн-зеркало в дискорде") == FORMAT_ONLINE


def test_venue_level_format_inference() -> None:
    assert is_online_venue("Онлайн / Гончар Денис") is True
    assert is_online_venue("Discord / Сервер игры") is True
    assert is_online_venue("Площадка Zoom") is True
    assert is_online_venue("Москва / Дудкина Алёна") is False
    assert is_online_venue("Одесса") is False
    assert is_online_venue(None) is False

    # Sync package played online vs in club
    assert infer_participation_format(FORMAT_SYNC, "Онлайн / Гончар Денис") == FORMAT_ONLINE
    assert infer_participation_format(FORMAT_SYNC, "Одесса") == FORMAT_SYNC
    assert infer_participation_format(FORMAT_ONLINE, "Одесса") == FORMAT_ONLINE
    assert infer_participation_format(FORMAT_OFFLINE, None) == FORMAT_OFFLINE


def test_aggregate_tournament_filtering() -> None:
    assert is_aggregate_tournament("Общий зачёт", "Общий зачёт сезона") is True
    assert is_aggregate_tournament("Сезон", "Студенческий сезон 2023") is True
    assert is_aggregate_tournament("Рейтинг", "Сводная таблица") is True
    assert is_aggregate_tournament("Очный", "Чемпионат Мира 2023") is False


def test_synthetic_dataset_generation(tmp_path: Path) -> None:
    pq_path = tmp_path / "test_synthetic.parquet"
    df = generate_synthetic_dataset(
        output_parquet_path=pq_path,
        num_tournaments=3,
        num_players=20,
        num_teams=5,
        questions_per_tournament=12,
    )

    assert pq_path.exists()
    assert not df.empty
    expected_cols = {
        "tournament_id",
        "tournament_type",
        "date",
        "team_id",
        "player_ids",
        "question_id",
        "solved",
    }
    assert expected_cols.issubset(df.columns)

    # Validate solved is 0 or 1
    assert set(df["solved"].unique()).issubset({0, 1})

    # Validate player rosters are lists of up to 6 players
    for p_list in df["player_ids"]:
        assert len(p_list) <= 6
        assert len(p_list) > 0


@pytest.mark.anyio
async def test_update_incremental(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.db"
    parquet_path = tmp_path / "dataset.parquet"

    # Create initial dataset with tournament 101 and 102
    initial_df = pd.DataFrame([
        {
            "tournament_id": 101,
            "tournament_type": FORMAT_OFFLINE,
            "date": "2024-01-01",
            "team_id": 1,
            "player_ids": [10, 11],
            "question_id": "101_1",
            "solved": 1,
        },
        {
            "tournament_id": 102,
            "tournament_type": FORMAT_SYNC,
            "date": "2024-01-02",
            "team_id": 2,
            "player_ids": [20, 21],
            "question_id": "102_1",
            "solved": 0,  # Old answer
        },
    ])
    initial_df.to_parquet(parquet_path, index=False)

    config = DataIngestConfig(
        cache_db_path=cache_path,
        parquet_path=parquet_path,
        max_requests_per_second=10.0,
    )
    loader = ChGKDataIngester(config)

    # Mock fetch_tournaments to return tournament 102 (edited) and 103 (brand new)
    mock_tournaments = [
        {"id": 102, "name": "Sync 102 Updated", "type": {"name": "Синхрон"}, "dateStart": "2024-01-02"},
        {"id": 103, "name": "Online 103 New", "type": {"name": "Онлайн"}, "dateStart": "2024-01-03"},
    ]

    # Mock results: 102 now has solved=1 (appeal accepted), 103 has solved=1
    def mock_results(client, t_id):
        if t_id == 102:
            return [
                {
                    "team": {"id": 2},
                    "teamMembers": [{"player": {"id": 20}}, {"player": {"id": 21}}],
                    "mask": "1",
                }
            ]
        elif t_id == 103:
            return [
                {
                    "team": {"id": 3},
                    "teamMembers": [{"player": {"id": 30}}],
                    "mask": "1",
                }
            ]
        return []

    from unittest.mock import AsyncMock, patch

    with patch.object(loader, "fetch_tournaments", new_callable=AsyncMock) as mock_fetch_t, \
         patch.object(loader, "fetch_tournament_results", new_callable=AsyncMock) as mock_fetch_res:
        mock_fetch_t.return_value = mock_tournaments
        mock_fetch_res.side_effect = mock_results

        merged = await loader.update_incremental(since_date="2024-01-02")

        assert len(merged) == 3
        # Tournaments 101, 102, 103
        assert set(merged["tournament_id"]) == {101, 102, 103}

        # Verify tournament 102 was updated (solved is now 1 instead of 0)
        row_102 = merged[merged["tournament_id"] == 102].iloc[0]
        assert row_102["solved"] == 1

        # Verify tournament 101 remained unchanged
        row_101 = merged[merged["tournament_id"] == 101].iloc[0]
        assert row_101["solved"] == 1

        # Verify sync state was updated
        assert loader.cache.get_sync_state("rating_last_sync") is not None
