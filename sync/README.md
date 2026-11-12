# sync

App Engine Standard F1: каждый день дописывает обновления турнирного сайта и базы вопросов в Cloud Storage и принимает вебхук управляющего Telegram-бота.

Расчёт рейтинга — корень этого репозитория. Разбор текстов, авторов и пакетов — `../questions`. Сюда не входят PyTorch и корпус вопросов: ночной cron на F1 пишет только дельту, снимок уже лежит в бакете.

| Объект | Что это | Когда |
| :--- | :--- | :--- |
| `questions/snapshot.parquet` | тексты, ответы, авторы | заливается один раз |
| `questions/parts/*.jsonl.gz` | пакеты, у которых изменился текст | 03:30 Europe/Berlin |
| `rating/snapshot.parquet` | результаты команд по вопросам | заливается один раз |
| `rating/parts/*.jsonl.gz` | турниры, поправленные за последние дни | 04:00 Europe/Berlin, дальше Cloud Tasks |
| `rating/credits.jsonl.gz` | редакторы туров | вместе с проходом рейтинга |
| `state/*.json` | курсоры | перезаписываются на месте |

## Локально

```bash
uv sync
uv run pytest
uv run python cli.py status
```

`uv sync` ставит и локальные пакеты (pandas, httpx, click). В `requirements.txt`, который читает App Engine, их нет: после правки `pyproject.toml` пересоберите его из корня этого проекта.

```bash
uv lock
uv export --no-dev --no-hashes --no-emit-project -o requirements.txt
```

Снимок заливается с машины, где parquet уже есть. Ночной cron его не скачивает.

```bash
uv run python cli.py upload-questions --path ../questions/data/gotquestions_questions.parquet
uv run python cli.py upload-rating --path ../data/dataset.parquet
uv run python cli.py materialize-data
```

`materialize-data` собирает `../questions/data/gotquestions_questions.parquet`, `../data/dataset.parquet` и `../data/rating_credits.parquet`.

Секреты: `cp env_template.yaml env.yaml` и заполнить `PROJECT_ID`, `TELEGRAM_API_TOKEN`, `OBFUSCATION_TOKEN`, `ADMIN_CHAT_ID`, `GCS_BUCKET`. `env.yaml` не коммитить. Пока `ADMIN_CHAT_ID` пуст, `/start` отвечает chat id и больше ничего не делает.

## Деплой

`gcloud app deploy` очередь не создаёт. Сервис берёт уже существующую: `CLOUD_TASKS_QUEUE` и `APP_REGION` из `env.yaml`, по умолчанию `bot-tasks` в `us-central1`. Команда `queues create` нужна только если в этом проекте Google Cloud такой очереди ещё нет. Повторный запуск завершится ошибкой «already exists».

```bash
gcloud config set project PROJECT_ID
gcloud app describe >/dev/null 2>&1 || gcloud app create --region=us-central
gcloud services enable datastore.googleapis.com cloudtasks.googleapis.com
# один раз, если очереди ещё нет:
gcloud tasks queues create bot-tasks --location=us-central1

gcloud storage buckets create gs://PROJECT_ID-chgk-data \
  --location=us-central1 --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding gs://PROJECT_ID-chgk-data \
  --member=serviceAccount:PROJECT_ID@appspot.gserviceaccount.com \
  --role=roles/storage.objectAdmin

gcloud artifacts repositories set-cleanup-policies gae-standard \
  --location=us-central1 \
  --project=PROJECT_ID \
  --policy=artifact_registry_policy.json \
  --no-dry-run

gcloud app deploy app.yaml cron.yaml
curl "https://PROJECT_ID.appspot.com/setwebhook"
```

`APP_REGION` в `env.yaml` — регион очереди Cloud Tasks (`us-central1`), и он должен совпадать с очередью. Если приложение App Engine уже создано в другом регионе, новое не создавать: регион выбирается один раз.

Проверка: `curl https://PROJECT_ID.appspot.com/health` и `curl https://PROJECT_ID.appspot.com/botinfo`.

Локальный вебхук: `uv run python main.py` (порт 8080). Datastore локально не нужен.
