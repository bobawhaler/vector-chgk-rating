"""Filter the local question table. No network."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_WHERE = {"text", "answer", "comment", "source", "all"}
_TEXT_COLUMNS = ("text", "answer", "comment", "source", "zachet", "nezachet")
_LIST_COLUMNS = (
    "author_names",
    "author_player_ids",
    "editor_names",
    "editor_player_ids",
    "pack_editor_names",
    "pack_editor_player_ids",
)


class QueryError(ValueError):
    """The filter itself is not usable."""


def _check_day(value: str, label: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if _DAY.fullmatch(text) is None:
        raise QueryError(f"{label}: нужна дата в виде ГГГГ-ММ-ДД.")
    return text


def _check_id(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise QueryError(f"{label}: нужно положительное целое.")
    return value


@dataclass(frozen=True)
class QuestionFilter:
    """All fields are combined with AND. A person matches as a substring."""

    text: str = ""
    where: str = "text"
    author: str = ""
    author_id: int | None = None
    editor: str = ""
    editor_id: int | None = None
    pack: str = ""
    pack_id: int | None = None
    tournament_id: int | None = None
    tour: int | None = None
    number: int | None = None
    since: str = ""
    until: str = ""
    limit: int = 20
    offset: int = 0
    unique: bool = True

    def __post_init__(self) -> None:
        where = (self.where or "text").strip().lower()
        if where not in _WHERE:
            raise QueryError("Поле поиска: text, answer, comment, source или all.")
        if self.limit < 1 or self.limit > 200:
            raise QueryError("limit: число от 1 до 200.")
        if self.offset < 0:
            raise QueryError("offset: число не меньше нуля.")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "author", self.author.strip())
        object.__setattr__(self, "editor", self.editor.strip())
        object.__setattr__(self, "pack", self.pack.strip())
        object.__setattr__(self, "since", _check_day(self.since, "since"))
        object.__setattr__(self, "until", _check_day(self.until, "until"))
        object.__setattr__(self, "author_id", _check_id(self.author_id, "author_id"))
        object.__setattr__(self, "editor_id", _check_id(self.editor_id, "editor_id"))
        object.__setattr__(self, "pack_id", _check_id(self.pack_id, "pack_id"))
        object.__setattr__(self, "tournament_id", _check_id(self.tournament_id, "tournament"))
        object.__setattr__(self, "tour", _check_id(self.tour, "tour"))
        object.__setattr__(self, "number", _check_id(self.number, "number"))


@dataclass(frozen=True)
class QueryPage:
    total: int
    offset: int
    limit: int
    rows: list[dict[str, object]]


def _as_list(value: object) -> list[object]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _contains_text(series: pd.Series, needle: str) -> pd.Series:
    folded = needle.casefold()

    def hit(value: object) -> bool:
        return any(folded in str(item).casefold() for item in _as_list(value) if str(item).strip())

    return series.map(hit)


def _contains_id(series: pd.Series, player_id: int) -> pd.Series:
    def hit(value: object) -> bool:
        for item in _as_list(value):
            try:
                if int(item) == player_id:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    return series.map(hit)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="object")
    return frame[name]


def _dates(frame: pd.DataFrame, name: str) -> pd.Series:
    return _column(frame, name).map(lambda value: value if isinstance(value, str) else "").fillna("").astype(str)


def _played_between(frame: pd.DataFrame, since: str, until: str) -> pd.Series:
    start = _dates(frame, "start_date")
    end = _dates(frame, "end_date")
    published = _dates(frame, "pub_date")
    left = start.where(start.str.len() > 0, end)
    left = left.where(left.str.len() > 0, published)
    right = end.where(end.str.len() > 0, start)
    right = right.where(right.str.len() > 0, published)
    mask = left.str.len() > 0
    if since:
        mask = mask & (right >= since)
    if until:
        mask = mask & (left <= until)
    return mask


def _people_mask(frame: pd.DataFrame, columns: tuple[str, ...], needle: str) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for name in columns:
        if name in frame.columns:
            mask = mask | _contains_text(frame[name], needle)
    return mask


def _id_mask(frame: pd.DataFrame, columns: tuple[str, ...], player_id: int) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for name in columns:
        if name in frame.columns:
            mask = mask | _contains_id(frame[name], player_id)
    return mask


def _text_mask(frame: pd.DataFrame, needle: str, where: str) -> pd.Series:
    columns = _TEXT_COLUMNS if where == "all" else (where,)
    mask = pd.Series(False, index=frame.index)
    for name in columns:
        if name not in frame.columns:
            continue
        mask = mask | frame[name].fillna("").astype(str).str.contains(needle, case=False, regex=False)
    return mask


def _plain(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return value
    return value


def _row(record: dict[str, object]) -> dict[str, object]:
    row = {key: _plain(value) for key, value in record.items()}
    for name in _LIST_COLUMNS:
        value = row.get(name, [])
        row[name] = value if isinstance(value, list) else []
    if row.get("number_in_tour") in ("", None):
        row["number_in_tour"] = row.get("question_number") or ""
    return row


def matched(questions: pd.DataFrame, spec: QuestionFilter) -> pd.DataFrame:
    """All questions matching ``spec``, newest packages first.

    ``limit`` and ``offset`` are ignored: those belong to :func:`run`.
    """
    frame = questions
    if frame.empty:
        return frame.iloc[0:0]

    mask = pd.Series(True, index=frame.index)
    if spec.text:
        mask = mask & _text_mask(frame, spec.text, spec.where)
    if spec.author:
        mask = mask & _people_mask(frame, ("author_names",), spec.author)
    if spec.author_id is not None:
        mask = mask & _id_mask(frame, ("author_player_ids",), spec.author_id)
    if spec.editor:
        mask = mask & _people_mask(frame, ("editor_names", "pack_editor_names"), spec.editor)
    if spec.editor_id is not None:
        mask = mask & _id_mask(frame, ("editor_player_ids", "pack_editor_player_ids"), spec.editor_id)
    if spec.pack:
        mask = mask & _column(frame, "pack_title").fillna("").astype(str).str.contains(spec.pack, case=False, regex=False)
    if spec.pack_id is not None and "pack_id" in frame.columns:
        mask = mask & (frame["pack_id"].fillna(0).astype(int) == spec.pack_id)
    if spec.tournament_id is not None and "rating_tournament_id" in frame.columns:
        mask = mask & (frame["rating_tournament_id"].fillna(0).astype(int) == spec.tournament_id)
    if spec.tour is not None and "tour_number" in frame.columns:
        mask = mask & (frame["tour_number"].fillna(0).astype(int) == spec.tour)
    if spec.number is not None:
        numbers = _column(frame, "number_in_tour")
        if "question_number" in frame.columns:
            numbers = numbers.where(numbers.notna() & (numbers.astype(str) != ""), frame["question_number"])
        mask = mask & (pd.to_numeric(numbers, errors="coerce").fillna(-1).astype(int) == spec.number)
    if spec.since or spec.until:
        mask = mask & _played_between(frame, spec.since, spec.until)

    found = frame.loc[mask]
    if found.empty:
        return found

    sort_columns = [name for name in ("start_date", "pack_id", "question_number") if name in found.columns]
    if "start_date" in found.columns:
        played = found["start_date"].fillna("").astype(str)
        found = found.assign(_played=played.where(played.str.len() > 0, "0000"))
        sort_columns = ["_played", *[name for name in sort_columns if name != "start_date"]]
    if sort_columns:
        ascending = [name not in ("_played", "pack_id") for name in sort_columns]
        found = found.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        if "_played" in found.columns:
            found = found.drop(columns=["_played"])
    if spec.unique and {"pack_id", "question_number"}.issubset(found.columns):
        found = found.drop_duplicates(["pack_id", "question_number"], keep="first")
    return found


def materialize(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Plain records for the rows of ``frame``, in their current order."""
    return [_row(record) for record in frame.to_dict("records")]


def run(questions: pd.DataFrame, spec: QuestionFilter) -> QueryPage:
    """Return a page of questions. The same package question is returned once."""
    found = matched(questions, spec)
    total = int(len(found))
    page = found.iloc[spec.offset:spec.offset + spec.limit]
    return QueryPage(total, spec.offset, spec.limit, materialize(page))
