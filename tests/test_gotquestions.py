"""Unit tests for gotquestions.online ingestion, parsing, caching, and export."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from chgk_rating.config import GotQuestionsConfig
from chgk_rating.gotquestions import (
    GotQuestionsCache,
    GotQuestionsIngester,
    extract_pack_from_rsc,
)

SAMPLE_PACK = {
    "id": 7099,
    "title": "Кубок Бесконечности. Финал",
    "pubDate": "2024-05-20",
    "tIds": [14031, 14032],
    "tours": [
        {
            "id": 101,
            "title": "Тур 1",
            "editors": [
                {"person": {"id": 999, "name": "Тестовый Редактор", "playerId": 155079}}
            ],
            "questions": [
                {
                    "number": 1,
                    "text": "Тестовый текст вопроса 1",
                    "answer": "Тестовый ответ 1",
                    "comment": "Тестовый комментарий 1",
                    "authors": [
                        {"person": {"id": 888, "name": "Тестовый Автор", "playerId": 12345}}
                    ],
                },
                {
                    "number": 2,
                    "text": "Тестовый текст вопроса 2",
                    "answer": "Тестовый ответ 2",
                    "comment": "",
                    "authors": [],
                },
            ],
        }
    ],
}


def test_extract_pack_from_rsc() -> None:
    # Test valid RSC stream line
    rsc_payload = '1:{"b":"c"}\n2:{"pack":' + json.dumps(SAMPLE_PACK) + "}\n3:null\n"
    extracted = extract_pack_from_rsc(rsc_payload)
    assert extracted is not None
    assert extracted["id"] == 7099
    assert extracted["title"] == "Кубок Бесконечности. Финал"
    assert len(extracted["tours"]) == 1

    # Test nested RSC stream line
    nested_payload = '1:{"root":{"slot":{"pack":' + json.dumps(SAMPLE_PACK) + "}}}\n"
    nested_extracted = extract_pack_from_rsc(nested_payload)
    assert nested_extracted is not None
    assert nested_extracted["id"] == 7099

    # Test invalid / non-pack response
    assert extract_pack_from_rsc("404: Not Found") is None
    assert extract_pack_from_rsc('1:{"other_data":123}') is None


def test_gotquestions_cache(tmp_path: Path) -> None:
    db_path = tmp_path / "gq_test.db"
    cache = GotQuestionsCache(db_path)

    # Initially empty
    assert cache.get_pack(7099) is None
    assert cache.is_known_not_found(7099) is False
    assert cache.get_max_pack_id() == 0
    assert cache.get_saved_pack_count() == 0

    # Save pack
    cache.save_pack(7099, SAMPLE_PACK)
    saved = cache.get_pack(7099)
    assert saved is not None
    assert saved["id"] == 7099
    assert cache.get_max_pack_id() == 7099
    assert cache.get_saved_pack_count() == 1

    # Mark not found
    cache.mark_not_found(7100)
    assert cache.is_known_not_found(7100) is True
    assert cache.get_pack(7100) is None

    # Saving over not found clears not found status
    cache.save_pack(7100, {"id": 7100, "title": "New Pack", "tIds": []})
    assert cache.is_known_not_found(7100) is False
    assert cache.get_pack(7100) is not None
    assert cache.get_saved_pack_count() == 2

    # Get all packs
    all_packs = cache.get_all_packs()
    assert len(all_packs) == 2
    assert all_packs[0]["id"] == 7099
    assert all_packs[1]["id"] == 7100


def test_export_to_parquet(tmp_path: Path) -> None:
    db_path = tmp_path / "gq_test.db"
    q_pq = tmp_path / "questions.parquet"
    p_pq = tmp_path / "packs.parquet"

    cache = GotQuestionsCache(db_path)
    cache.save_pack(7099, SAMPLE_PACK)

    config = GotQuestionsConfig(
        cache_db_path=db_path,
        questions_parquet=q_pq,
        packs_parquet=p_pq,
    )
    ingester = GotQuestionsIngester(config)

    q_df, p_df = ingester.export_to_parquet()

    assert p_pq.exists()
    assert q_pq.exists()
    assert len(p_df) == 1
    assert p_df.iloc[0]["pack_id"] == 7099
    assert p_df.iloc[0]["rating_tournament_ids"] == [14031, 14032]

    # Two rating tournament IDs * 2 questions = 4 rows
    assert len(q_df) == 4
    expected_qids = {"14031_1", "14031_2", "14032_1", "14032_2"}
    assert set(q_df["question_id"]) == expected_qids

    row_q1 = q_df[q_df["question_id"] == "14031_1"].iloc[0]
    assert row_q1["author_names"] == ["Тестовый Автор"]
    assert row_q1["author_player_ids"] == [12345]
    assert row_q1["editor_names"] == ["Тестовый Редактор"]
    assert row_q1["editor_player_ids"] == [155079]
    assert row_q1["text"] == "Тестовый текст вопроса 1"
    assert row_q1["answer"] == "Тестовый ответ 1"


@pytest.mark.anyio
async def test_ingest_incremental_mocked(tmp_path: Path) -> None:
    db_path = tmp_path / "gq_mock.db"
    config = GotQuestionsConfig(
        cache_db_path=db_path,
        consecutive_404_stop=3,
        max_requests_per_second=100.0,
    )
    ingester = GotQuestionsIngester(config)

    # Mock responses for pack 1 (200), pack 2 (200), pack 3 (404), pack 4 (404), pack 5 (404)
    async def mock_get(url: str, **kwargs):
        pack_id = int(str(url).split("/")[-1])
        if pack_id in (1, 2):
            content = '1:{"pack":' + json.dumps({"id": pack_id, "title": f"Pack {pack_id}", "tIds": []}) + "}\n"
            return httpx.Response(200, text=content, request=httpx.Request("GET", url))
        else:
            return httpx.Response(404, request=httpx.Request("GET", url))

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_client_get:
        mock_client_get.side_effect = mock_get
        fetched = await ingester.ingest_incremental(lookback_packs=0)

        assert fetched == 2
        assert ingester.cache.get_saved_pack_count() == 2
        assert ingester.cache.get_max_pack_id() == 2
        assert ingester.cache.is_known_not_found(3) is True
        assert ingester.cache.is_known_not_found(4) is True
        assert ingester.cache.is_known_not_found(5) is True
