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
```

Файлы по умолчанию:

| Файл | Что это |
| :--- | :--- |
| `data/gotquestions_questions.parquet` | вопросы |
| `data/gotquestions_packs.parquet` | пакеты |
| `data/gotquestions_cache.db` | локальный кэш полной выгрузки |
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
