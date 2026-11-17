"""The local page reads the same filters as the query function."""

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

from corpus.browse import make_server


def _serve(frame: pd.DataFrame):
    server = make_server(frame, {"questions": 1, "packs": 1, "span": "", "with_source": 0}, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def test_page_returns_the_question_with_its_source() -> None:
    frame = pd.DataFrame([{
        "question_id": "gq_1_1",
        "rating_tournament_id": 0,
        "pack_id": 1,
        "pack_title": "Пакет",
        "tour_number": 1,
        "question_number": 1,
        "number_in_tour": 1,
        "text": "Текст про кота",
        "answer": "кот",
        "comment": "комментарий",
        "source": "https://example.test/cat",
        "zachet": "",
        "nezachet": "",
        "start_date": "2024-01-02",
        "end_date": "2024-01-02",
        "pub_date": "",
        "author_names": ["Анна"],
        "author_player_ids": [3],
        "editor_names": ["Борис"],
        "editor_player_ids": [4],
        "pack_editor_names": [],
        "pack_editor_player_ids": [],
    }])
    server, base = _serve(frame)
    try:
        page = urlopen(base + "/", timeout=5)
        assert page.status == 200
        assert "База вопросов" in page.read().decode()
        found = json.load(urlopen(base + "/api/questions?author=" + quote("Анна"), timeout=5))
        assert found["total"] == 1
        assert found["rows"][0]["answer"] == "кот"
        assert found["rows"][0]["source"] == "https://example.test/cat"
    finally:
        server.shutdown()
        server.server_close()


def test_bad_date_returns_400() -> None:
    server, base = _serve(pd.DataFrame())
    try:
        try:
            urlopen(base + "/api/questions?since=" + quote("вчера"), timeout=5)
        except HTTPError as exc:
            assert exc.code == 400
            payload = json.load(exc)
            assert "ГГГГ-ММ-ДД" in payload["error"]
        else:
            raise AssertionError("ожидался ответ 400")
    finally:
        server.shutdown()
        server.server_close()
