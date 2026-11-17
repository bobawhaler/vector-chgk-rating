"""Turn a question sample into a prompt. This module does not call a model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from corpus.query import materialize

DEFAULT_REVEAL = ("text", "answer", "zachet", "nezachet", "comment", "source")

_LABELS = {
    "text": "Вопрос",
    "answer": "Ответ",
    "zachet": "Зачёт",
    "nezachet": "Незачёт",
    "comment": "Комментарий",
    "source": "Источник",
    "author_names": "Авторы",
    "editor_names": "Редакторы",
    "pack_editor_names": "Редакторы пакета",
    "pack_title": "Пакет",
    "start_date": "Дата",
    "end_date": "Окончание",
    "pub_date": "Опубликован",
}

JSON_CONTRACT = (
    "Верни только JSON-массив и никакого другого текста. "
    "Каждый элемент — объект с полем \"id\", равным id вопроса из метки [id=...]. "
    "Остальные поля определи по инструкции."
)


@dataclass(frozen=True)
class Prompt:
    """One request to send to a model, and the question ids it covers."""

    instruction: str
    text: str
    ids: tuple[str, ...]
    reveal: tuple[str, ...]


def _rows(source: object) -> list[dict[str, object]]:
    if isinstance(source, pd.DataFrame):
        return materialize(source)
    boxed = getattr(source, "rows", None)
    if boxed is not None and not isinstance(source, (str, bytes, dict)):
        return [dict(row) for row in boxed]
    if isinstance(source, dict):
        raise TypeError("Нужен набор вопросов, не одна запись.")
    return [dict(row) for row in source]  # type: ignore[arg-type]


def _value(row: dict[str, object], name: str) -> str:
    value = row.get(name, "")
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def _block(row: dict[str, object], reveal: Sequence[str]) -> str:
    question_id = str(row.get("question_id") or "").strip()
    if not question_id:
        raise ValueError("У вопроса нет question_id.")
    lines = [f"[id={question_id}]"]
    for name in reveal:
        text = _value(row, name)
        if text:
            lines.append(f"{_LABELS[name]}: {text}")
    return "\n".join(lines)


def _check(instruction: str, reveal: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    task = instruction.strip()
    if not task:
        raise ValueError("Инструкция промпта пустая.")
    unknown = [name for name in reveal if name not in _LABELS]
    if unknown:
        allowed = ", ".join(sorted(_LABELS))
        raise ValueError(f"Неизвестные поля промпта: {', '.join(unknown)}. Доступны: {allowed}.")
    return task, tuple(reveal)


def _compose(task: str, rows: Sequence[dict[str, object]], reveal: Sequence[str], contract: bool) -> str:
    parts = [task]
    if contract:
        parts.append(JSON_CONTRACT)
    if rows:
        parts.append("\n\n".join(_block(row, reveal) for row in rows))
    return "\n\n".join(parts)


def render(
    source: object,
    instruction: str,
    *,
    reveal: Sequence[str] = DEFAULT_REVEAL,
    contract: bool = True,
) -> Prompt:
    """Build one prompt. By default the model sees the question, answer, comment and source.

    Pass ``reveal=("text",)`` when the model must not see the answer. The reply
    contract asks for a JSON array with an ``id`` per question, so the reply can
    be joined back onto the sample.
    """
    task, fields = _check(instruction, reveal)
    rows = _rows(source)
    return Prompt(
        instruction=task,
        text=_compose(task, rows, fields, contract),
        ids=tuple(str(row.get("question_id") or "") for row in rows),
        reveal=fields,
    )


def each(
    source: object,
    instruction: str,
    *,
    reveal: Sequence[str] = DEFAULT_REVEAL,
    contract: bool = True,
) -> list[Prompt]:
    """One prompt per question, for separate model calls."""
    return [render([row], instruction, reveal=reveal, contract=contract) for row in _rows(source)]


def batches(
    source: object,
    instruction: str,
    *,
    max_chars: int = 12000,
    reveal: Sequence[str] = DEFAULT_REVEAL,
    contract: bool = True,
) -> list[Prompt]:
    """Split a sample into prompts that stay within ``max_chars``."""
    if max_chars < 1:
        raise ValueError("max_chars: нужно положительное число.")
    task, fields = _check(instruction, reveal)
    rows = _rows(source)
    if not rows:
        return []
    chunks: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for row in rows:
        trial = [*current, row]
        if current and len(_compose(task, trial, fields, contract)) > max_chars:
            chunks.append(current)
            current = [row]
        else:
            current = trial
    chunks.append(current)
    return [render(chunk, task, reveal=fields, contract=contract) for chunk in chunks]
