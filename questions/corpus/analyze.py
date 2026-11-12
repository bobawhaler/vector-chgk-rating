"""Summaries over the question parquet. No network."""

from __future__ import annotations

import pandas as pd


def _names(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns or frame.empty:
        return pd.Series(dtype="object")
    names = frame[column].explode()
    names = names.dropna().astype(str).str.strip()
    return names[names.str.len() > 0]


def summarize(questions: pd.DataFrame, packs: pd.DataFrame | None = None) -> str:
    """Counts of questions, packages, authors, and editors."""
    if questions.empty:
        return "В таблице вопросов нет строк."
    linked = questions
    if "rating_tournament_id" in questions.columns:
        linked = questions[questions["rating_tournament_id"].fillna(0).astype(int) > 0]
    packs_in_questions = questions["pack_id"].nunique() if "pack_id" in questions.columns else 0
    lines = [
        f"Вопросов: {len(questions)}",
        f"Пакетов в вопросах: {packs_in_questions}",
        f"С привязкой к турниру рейтинга: {len(linked)}",
        f"Авторов: {_names(questions, 'author_names').nunique()}",
        f"Редакторов туров: {_names(questions, 'editor_names').nunique()}",
    ]
    if packs is not None and not packs.empty:
        lines.insert(2, f"Пакетов в таблице пакетов: {len(packs)}")
    return "\n".join(lines)


def people(questions: pd.DataFrame, column: str, limit: int = 20) -> pd.DataFrame:
    """How many question rows name each person in ``column``."""
    counts = _names(questions, column).value_counts().head(limit)
    return counts.rename_axis("name").reset_index(name="questions")


def link_coverage(rating: pd.DataFrame, questions: pd.DataFrame) -> str:
    """How many rating question ids have a text in the corpus."""
    if "question_id" not in rating.columns or "question_id" not in questions.columns:
        return "В одной из таблиц нет question_id."
    rating_ids = set(rating["question_id"].astype(str))
    linked = questions
    if "rating_tournament_id" in questions.columns:
        linked = questions[questions["rating_tournament_id"].fillna(0).astype(int) > 0]
    question_ids = set(linked["question_id"].astype(str))
    common = rating_ids & question_ids
    coverage = len(common) / max(len(rating_ids), 1) * 100.0
    return "\n".join([
        f"Уникальных вопросов в рейтинге: {len(rating_ids)}",
        f"Вопросов корпуса с турниром рейтинга: {len(question_ids)}",
        f"Совпало по question_id: {len(common)} ({coverage:.2f}%)",
    ])


def search(questions: pd.DataFrame, text: str, limit: int = 20) -> pd.DataFrame:
    """Questions whose text contains ``text``, case-insensitive."""
    if "text" not in questions.columns or not text.strip():
        return questions.iloc[0:0]
    mask = questions["text"].fillna("").astype(str).str.contains(text.strip(), case=False, regex=False)
    columns = [name for name in ("question_id", "pack_id", "question_number", "text", "answer") if name in questions.columns]
    return questions.loc[mask, columns].head(limit)
