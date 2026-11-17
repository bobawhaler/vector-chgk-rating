"""Local page for reading questions from the catalog."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from corpus.query import QueryError, QuestionFilter, run

_PAGE = Path(__file__).with_name("browse.html").read_bytes()


def _one(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0].strip() if values else ""


def _optional_int(params: dict[str, list[str]], key: str) -> int | None:
    raw = _one(params, key)
    if not raw:
        return None
    if not raw.isdigit():
        raise QueryError(f"{key}: нужно целое число.")
    return int(raw)


def filter_from_query(params: dict[str, list[str]]) -> QuestionFilter:
    limit_raw = _one(params, "limit") or "20"
    offset_raw = _one(params, "offset") or "0"
    if not limit_raw.isdigit() or not offset_raw.isdigit():
        raise QueryError("limit и offset: нужны целые числа.")
    duplicates = _one(params, "duplicates").lower() in {"1", "true", "on", "yes"}
    return QuestionFilter(
        text=_one(params, "text"),
        where=_one(params, "where") or "text",
        author=_one(params, "author"),
        author_id=_optional_int(params, "author_id"),
        editor=_one(params, "editor"),
        editor_id=_optional_int(params, "editor_id"),
        pack=_one(params, "pack"),
        pack_id=_optional_int(params, "pack_id"),
        tournament_id=_optional_int(params, "tournament"),
        tour=_optional_int(params, "tour"),
        number=_optional_int(params, "number"),
        since=_one(params, "since"),
        until=_one(params, "until"),
        limit=int(limit_raw),
        offset=int(offset_raw),
        unique=not duplicates,
    )


def _send(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _json(handler: BaseHTTPRequestHandler, status: int, payload: object) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode()
    _send(handler, status, body, "application/json; charset=utf-8")


def handle(handler: BaseHTTPRequestHandler, questions: pd.DataFrame, meta: dict[str, object]) -> None:
    parsed = urlparse(handler.path)
    if parsed.path == "/":
        _send(handler, 200, _PAGE, "text/html; charset=utf-8")
        return
    if parsed.path == "/api/meta":
        _json(handler, 200, meta)
        return
    if parsed.path != "/api/questions":
        _json(handler, 404, {"error": "Нет такой страницы."})
        return
    try:
        spec = filter_from_query(parse_qs(parsed.query, keep_blank_values=False))
        page = run(questions, spec)
    except QueryError as exc:
        _json(handler, 400, {"error": str(exc)})
        return
    _json(handler, 200, {
        "total": page.total,
        "offset": page.offset,
        "limit": page.limit,
        "rows": page.rows,
    })


def make_server(
    questions: pd.DataFrame,
    meta: dict[str, object],
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    table = questions
    summary = meta

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            handle(self, table, summary)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    questions: pd.DataFrame,
    meta: dict[str, object],
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = make_server(questions, meta, host, port)
    host_name, bound_port = server.server_address[:2]
    print(f"http://{host_name}:{bound_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
