"""What the corpus knows about one realia.

Questions that mention a name, the other realia named in those questions or
given as their answer, the editors who set the name, and the editors who set
its neighborhood. No model is called. A common answer such as «Марс» joins
the neighborhood only together with the name or with another, rarer realia
from the same questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from corpus.query import QuestionFilter, matched

_BROAD = 40
_COLUMNS = ("text", "answer", "comment", "zachet")
_EDGE = re.compile(r"^[\s.,;:!?…]+|[\s.,;:!?…]+$")
_QUOTED = re.compile(r"«([^»]{2,80})»|\"([^\"]{2,80})\"")
_CAPITAL = re.compile(r"[А-ЯЁA-Z][а-яёa-zA-Z-]{2,}(?:\s+[А-ЯЁA-Z][а-яёa-zA-Z-]{2,})*")
_PLACEHOLDER = frozenset({
    "икс", "альфа", "пропуск", "первый", "вторая", "второе", "второй",
    "он", "она", "они", "оно", "такой", "такая", "такие",
})
_OPENING = frozenset({
    "назовите", "какой", "какая", "какие", "какого", "кто", "что", "где",
    "когда", "почему", "как", "ответьте", "внимание", "согласно", "перед",
    "после", "один", "одна", "это", "этот", "эта", "дуплет", "блиц",
    "вопрос", "заполните", "напишите", "укажите",
})


class KnowledgeError(ValueError):
    """The name cannot be looked up."""


@dataclass(frozen=True)
class Mention:
    question_id: str
    date: str
    pack: str
    role: str
    answer: str
    named: tuple[str, ...]
    editors: tuple[str, ...]
    known: str


@dataclass(frozen=True)
class Link:
    name: str
    questions: int


@dataclass(frozen=True)
class EditorUse:
    name: str
    questions: int
    realia: tuple[str, ...]


@dataclass(frozen=True)
class Card:
    name: str
    mentions: tuple[Mention, ...]
    links: tuple[Link, ...]
    editors: tuple[EditorUse, ...]
    similar_editors: tuple[EditorUse, ...]


def about(frame: pd.DataFrame, name: str, *, broad_at: int = _BROAD) -> Card:
    """Collect the corpus card for one written form of a realia."""
    needle = name.strip()
    if not needle:
        raise KnowledgeError("Пустое имя реалии.")
    if broad_at < 2:
        raise KnowledgeError("broad_at: нужно число от 2.")
    found = matched(frame, QuestionFilter(text=needle, where="all", limit=1))
    mentions = tuple(_mention(row, needle) for row in found.to_dict("records"))
    links = _links(mentions, needle)
    editors = _editors_of(mentions, needle)
    similar = _similar(frame, needle, links, broad_at) if mentions else ()
    return Card(needle, mentions, links, editors, similar)


def format_card(card: Card, *, mentions: int = 8, editors: int = 8) -> str:
    """Russian text of a card. Counts are the full card, lines are capped."""
    lines = [
        f"{card.name}: вопросов {len(card.mentions)}, связанных реалий {len(card.links)}",
    ]
    if card.links:
        lines.append("Связи:")
        for link in card.links[:12]:
            lines.append(f"  {link.name} — {link.questions}")
    if card.editors:
        lines.append("Редакторы этой реалии:")
        for editor in card.editors[:editors]:
            lines.append(f"  {editor.name} — {editor.questions}")
    if card.similar_editors:
        lines.append("Редакторы похожих реалий:")
        for editor in card.similar_editors[:editors]:
            used = ", ".join(editor.realia[:6])
            lines.append(f"  {editor.name} — {len(editor.realia)} ({used})")
    if card.mentions:
        lines.append("Что сказано:")
        for mention in card.mentions[:mentions]:
            place = "ответ" if mention.role == "answer" else "в тексте"
            lines.append(f"  [{place}] {mention.answer}")
            if mention.known:
                lines.append(f"    {mention.known}")
    if not card.mentions:
        lines.append("В вопросах не встречается.")
    return "\n".join(lines)


def _mention(row: dict[str, object], needle: str) -> Mention:
    answer = _plain(row.get("answer"))
    text = _plain(row.get("text"))
    comment = _plain(row.get("comment"))
    role = "answer" if needle.casefold() in answer.casefold() else "clue"
    return Mention(
        str(row.get("question_id") or ""),
        _plain(row.get("start_date"))[:10],
        _plain(row.get("pack_title")),
        role,
        _label(answer) or answer.strip(),
        _explicit(text),
        _editors(row),
        _known(text, comment, needle),
    )


def _links(mentions: tuple[Mention, ...], needle: str) -> tuple[Link, ...]:
    counts: dict[str, int] = {}
    surface: dict[str, str] = {}
    folded_needle = needle.casefold()
    for mention in mentions:
        labels = list(mention.named)
        if mention.role == "clue":
            labels.append(mention.answer.strip())
        for label in labels:
            key = label.casefold()
            if len(key) < 3 or folded_needle in key or key in _PLACEHOLDER:
                continue
            counts[key] = counts.get(key, 0) + 1
            current = surface.get(key)
            if current is None or len(label) < len(current):
                surface[key] = label
    ranked = sorted(counts, key=lambda key: (-counts[key], surface[key].casefold()))
    return tuple(Link(surface[key], counts[key]) for key in ranked)


def _editors_of(mentions: tuple[Mention, ...], needle: str) -> tuple[EditorUse, ...]:
    questions: dict[str, set[str]] = {}
    for mention in mentions:
        for name in mention.editors:
            questions.setdefault(name, set()).add(mention.question_id or mention.known)
    ranked = sorted(questions, key=lambda name: (-len(questions[name]), name.casefold()))
    return tuple(EditorUse(name, len(questions[name]), (needle,)) for name in ranked)


def _similar(
    frame: pd.DataFrame,
    needle: str,
    links: tuple[Link, ...],
    broad_at: int,
) -> tuple[EditorUse, ...]:
    if frame.empty or not links:
        return ()
    unique = frame
    if {"pack_id", "question_number"}.issubset(frame.columns):
        unique = frame.drop_duplicates(["pack_id", "question_number"], keep="first")
    blob = _blob(unique)
    answers = unique["answer"].fillna("").astype(str) if "answer" in unique.columns else blob
    query = _contains(blob, needle, stem=True)
    specific: list[str] = []
    broad: list[str] = []
    answer_hit: dict[str, pd.Series] = {}
    for link in links:
        hit = _contains(answers, link.name)
        answer_hit[link.name] = hit
        (broad if int(hit.sum()) >= broad_at else specific).append(link.name)
    near = query.copy()
    company = query.copy()
    for link in specific:
        text_hit = _contains(blob, link)
        near = near | answer_hit[link] | text_hit
        company = company | text_hit
    for link in broad:
        near = near | (answer_hit[link] & company)
    chosen = unique.loc[near]
    used: dict[str, set[str]] = {}
    seen: dict[str, set[str]] = {}
    for row in chosen.to_dict("records"):
        haystack = "\n".join(_plain(row.get(column)) for column in _COLUMNS)
        keys = {needle} if _has(haystack, needle, stem=True) else set()
        keys.update(link.name for link in links if _has(haystack, link.name))
        if not keys:
            continue
        question_id = str(row.get("question_id") or haystack[:40])
        for editor in _editors(row):
            used.setdefault(editor, set()).update(keys)
            seen.setdefault(editor, set()).add(question_id)
    ranked = sorted(
        (name for name, keys in used.items() if len(keys) >= 2),
        key=lambda name: (-len(used[name]), -len(seen[name]), name.casefold()),
    )
    return tuple(EditorUse(name, len(seen[name]), tuple(sorted(used[name], key=str.casefold))) for name in ranked)


def _blob(frame: pd.DataFrame) -> pd.Series:
    parts = [frame[column].fillna("").astype(str) for column in _COLUMNS if column in frame.columns]
    if not parts:
        return pd.Series("", index=frame.index)
    blob = parts[0]
    for part in parts[1:]:
        blob = blob.str.cat(part, sep="\n")
    return blob


def _editors(row: dict[str, object]) -> tuple[str, ...]:
    tour = _names(row.get("editor_names"))
    names = tour or _names(row.get("pack_editor_names"))
    return tuple(dict.fromkeys(names))


def _names(value: object) -> list[str]:
    if value is None or isinstance(value, str):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip() and str(item).strip() != "nan"]


def _plain(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text == "nan" else text


def _explicit(text: str) -> tuple[str, ...]:
    """Names written in the question: quotes, and capitals that are not the opening word."""
    found: list[str] = []
    for match in _QUOTED.finditer(text):
        label = _label(next(group for group in match.groups() if group))
        if label:
            found.append(label)
    for match in _CAPITAL.finditer(text):
        phrase = match.group(0)
        at = match.start()
        before = text[:at].rstrip()
        sentence_start = not before or before[-1] in ".!?"
        words = phrase.split()
        if sentence_start and len(words) == 1:
            continue
        if words[0].casefold() in _OPENING or words[0].casefold() in _PLACEHOLDER:
            words = words[1:]
        if not words or _PLACEHOLDER.issuperset(word.casefold() for word in words):
            continue
        found.append(" ".join(words))
    unique: dict[str, str] = {}
    for label in found:
        unique.setdefault(label.casefold(), label)
    return tuple(unique.values())


def _label(answer: str) -> str:
    text = " ".join(answer.replace("[", " ").replace("]", " ").split())
    return _EDGE.sub("", text)


def _pattern(needle: str, *, stem: bool) -> str:
    if stem:
        tail = r"\w*"
    elif re.search(r"[а-яё]$", needle, flags=re.IGNORECASE):
        tail = r"(?:[а-яё]{0,3})?(?!\w)"
    else:
        tail = r"(?!\w)"
    return r"(?<!\w)" + re.escape(needle) + tail


def _contains(series: pd.Series, needle: str, *, stem: bool = False) -> pd.Series:
    return series.str.contains(_pattern(needle, stem=stem), case=False, regex=True)


def _has(haystack: str, needle: str, *, stem: bool = False) -> bool:
    return re.search(_pattern(needle, stem=stem), haystack, flags=re.IGNORECASE) is not None


def _known(text: str, comment: str, needle: str) -> str:
    folded = needle.casefold()
    for blob in (comment, text):
        at = blob.casefold().find(folded)
        if at < 0:
            continue
        start = max(0, at - 70)
        end = min(len(blob), at + len(needle) + 110)
        piece = " ".join(blob[start:end].split())
        if start:
            piece = "…" + piece
        if end < len(blob):
            piece = piece + "…"
        return piece
    return ""
