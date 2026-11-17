# questions

Разбор базы вопросов ЧГК: тексты, ответы, авторы и редакторы туров из gotquestions.online.

Ежедневную дельту качает `../sync` (App Engine и вебхук бота) и складывает её в Cloud Storage. Этот каталог читает локальный parquet и считает покрытие, авторов и редакторов. Обучение рейтинга здесь не запускается: таблица результатов лежит в `../data`.

```bash
uv sync
uv run pytest
uv run python cli.py summary
uv run python cli.py link
uv run python cli.py authors --limit 20
uv run python cli.py search "кот"
uv run python cli.py query --author "Иванов" --since 2024-01-01 --until 2024-12-31
uv run python cli.py browse
```

`query` и `browse` читают локальный каталог. В нём тексты берутся из актуального parquet, а источники, зачёт, даты отыгрыша и названия пакетов — из `data/gotquestions_cache.db`, если в самом parquet этих полей ещё нет. Каталог собирается сам при первом запросе и лежит в `data/gotquestions_catalog.parquet`. Страница открывается на `http://127.0.0.1:8765`.

Фильтры складываются: автор, id автора, редактор тура или пакета, id редактора, название или id пакета, id турнира рейтинга, номер тура, номер вопроса в туре, интервал дат отыгрыша и подстрока в тексте, ответе, комментарии или источнике. Без фильтров показываются самые свежие пакеты. Один и тот же вопрос пакета не повторяется на каждый id турнира; `--duplicates` оставляет копии.

Для разбора выборки моделью есть три модуля. Они не вызывают модель: выборка собирается локально, промпт уходит наружу, ответ модели возвращается сюда и стыкуется по `id`. `QuestionFilter.limit` ограничивает только страницу `query` и `browse`. Размер выборки задаёт `size`.

```python
from corpus.catalog import load_catalog
from corpus.prompt import render
from corpus.query import QuestionFilter
from corpus.results import analyze, coverage, tally
from corpus.sample import take

sample = take(
    load_catalog(),
    QuestionFilter(editor="Полевой", since="2024-01-01", until="2024-12-31"),
    size=30,
    seed=1,
    by="author",
)
prompt = render(sample, "Назови тему каждого вопроса одним словом в поле theme.")
# prompt.text — запрос к модели. Ответ модели:
report = analyze(prompt, model_reply)
coverage(report)
tally(report, "theme")
```

`by` берёт вопросы по очереди из групп `author`, `editor`, `year` или `pack`, чтобы одна группа не заняла всю выборку. `reveal=("text",)` прячет ответ, комментарий и источник. `batches` режет выборку по длине промпта, `each` делает отдельный промпт на вопрос. `same_text` считает точные совпадения поля ответа модели с полем вопроса после нормализации регистра и пробелов.

Порядок разбора пакета и одного вопроса — в [GUIDE.md](GUIDE.md). Классы ходов для разметки вопросов — в [MOVES.md](MOVES.md). У вопроса упорядоченный список шагов взятия. Приёмы редактора — другой тип сущности, след сборки фразы, они в [TECHNIQUES.md](TECHNIQUES.md). Реалии, свойства, факты и связи, которые ходы читают, — в [REALIA.md](REALIA.md). Куда складывается разбор каждого вопроса и как доставать его по редактору, теме и форме — в [STORE.md](STORE.md), модуль `corpus.store`. Инструкция для модели и проверка разбора — `corpus.realia`. Карточка одной реалии по текстам каталога — `corpus.knowledge.about` и команда `about`. Эти модули модель не вызывают.

Файлы по умолчанию:

| Файл | Что это |
| :--- | :--- |
| `data/gotquestions_questions.parquet` | вопросы, как их отдал sync |
| `data/gotquestions_packs.parquet` | пакеты |
| `data/gotquestions_cache.db` | локальный кэш полной выгрузки, в нём источники и даты |
| `data/gotquestions_catalog.parquet` | таблица для `query` и `browse` |
| `data/knowledge.sqlite` | разборы вопросов, когда их туда положили |
| `../data/dataset.parquet` | результаты, только для команды `link` |

Обновить локальный parquet из бакета:

```bash
cd ../sync
uv run python cli.py materialize-questions
```

Полная вежливая выгрузка с сайта (не ночной cron, около одного запроса в секунду):

```bash
uv run python cli.py fetch --lookback-packs 20 --rate-limit 1
```

После неё снимок в бакет заливает sync: `uv run python cli.py upload-questions`.
