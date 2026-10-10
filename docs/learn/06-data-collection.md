# 06. Сбор данных по расписанию (`src/collector/`, Celery + Redis)

## Что это делает
Раз в N часов (по умолчанию 6) система сама обходит список адресов (`data/sources.json`): скачивает
веб-страницы и PDF, чистит текст, отбрасывает дубликаты и добавляет **только новое** в базу знаний.

**Аналогия.** Библиотекарь с расписанием. Он ходит по списку источников, но: сначала смотрит на табличку
«сюда не входить» (robots.txt), не звонит в один и тот же дом чаще чем раз в 5 секунд, выкидывает рекламу
и меню из страницы, а потом проверяет по «отпечатку» (хэшу), нет ли уже такой книги на полке. Если
страница изменилась, старый экземпляр заменяется новым.

## Схема
```
Celery beat (расписание, каждые 360 мин)
   |  кладёт задачу в Redis
   v
collect_all -> по задаче collect_one на каждый URL (повторяются независимо)
   |
   v
robots.txt разрешает? --нет--> status=skipped (повторять бессмысленно)
   |да
   v
пауза (не чаще 1 запроса на сайт в 5 с) -> скачать (HTML или PDF, до 15 МБ)
   |
   v
очистка: убрать script/nav/footer -> NFKC -> пробелы -> хэш SHA-256 нормализованного текста
   |
   v
хэш уже есть?  --да--> duplicate (ничего не делаем)
   |нет
   v
URL уже собирали (текст изменился)? -> удалить старые фрагменты
   |
   v
нарезать на фрагменты -> добавить в хранилище (Chroma/pgvector) -> записать в collected_documents
```

## Файл за файлом
- `fetch.py`: `RateLimiter` (интервал на хост), `is_allowed` (robots.txt: 403 -> «нельзя», 404 -> «можно»),
  `fetch` (только HTML/PDF, лимит размера; ошибки HTTP поднимаются, чтобы Celery мог повторить).
- `clean.py`: `html_to_text` (bs4), `pdf_to_text` (pdfplumber), `normalize_text`, `content_hash`
  (хэш не зависит от регистра и пробелов).
- `pipeline.py`: `collect_url`: вся цепочка, идемпотентная. Статусы: `ingested`, `duplicate`, `updated`, `empty`.
- `tasks.py`: приложение Celery и задачи. `acks_late` (задача подтверждается после выполнения, поэтому
  упавший воркер не теряет её), `autoretry_for=RequestException` с экспоненциальной паузой и jitter,
  `max_retries=3`, `rate_limit`. Робота-запрет и неподдерживаемый тип считаются постоянными и не повторяются.
- `docker-compose.yml`: сервисы `redis`, `postgres`, `worker`, `beat` (профиль `full`).

## Как запустить
```bash
docker compose --profile full up -d postgres redis
celery -A src.collector.tasks worker --pool=solo --loglevel=info      # Windows: --pool=solo
python -c "from src.collector.tasks import collect_all; print(collect_all.delay().get())"
```

## Результат (честно)
Живой прогон на этой машине (реальный Redis + воркер): 2 страницы Wikipedia загружены (`ingested`), тестовый
PDF слишком короткий был пропущен (`empty`), **повторный запуск дал `duplicate` для обеих, число фрагментов не выросло**.
Тест с настоящим брокером и воркером Celery есть (`-m redis`, пропускается без Redis).

## Вопросы на собеседовании (EN)
1. **How do you make tasks idempotent?** A content hash is stored with a UNIQUE constraint; same text is skipped,
   chunk ids are stable so re-adding is an upsert.
2. **What does `acks_late` do?** The task is acknowledged after it finishes; if the worker dies, the broker redelivers it.
3. **Which errors are retried?** Network/HTTP errors with exponential backoff and jitter; robots blocks and bad content
   types are permanent, so they are reported but not retried.
4. **How do you stay polite?** robots.txt, a per-host minimum interval, an honest User-Agent, a Celery rate limit.
5. **Why hash the *normalized* text?** Whitespace or case changes should not create a "new" document.
6. **What if a page changes?** Same URL with a new hash: old chunks of that document are deleted, new ones added.
7. **Celery beat vs cron?** Beat keeps the schedule in the app config and runs tasks through the same broker and retries.
8. **Why one task per URL?** Failures and retries are independent; one slow site does not block the others.
9. **Why pgvector for the worker in Docker?** Chroma is not safe for two processes writing together.
10. **What are the legal/ethical limits?** Only public pages that allow crawling; respect licenses (Wikipedia is CC BY-SA).

## Ограничения
- Планировщик и воркер в Docker **не запускались у меня целиком**: проверены Redis и воркер Celery локально и конфигурация
  compose через `docker compose config`. Образ собирается (локально и в CI), но сами контейнеры `worker`/`beat` я не запускал.
- Лимит частоты работает в памяти одного процесса; для нескольких воркеров нужен общий (Redis) лимитер.
- Нет JavaScript-рендеринга: страницы, которым нужен браузер, не прочитаются.
- Дедупликация точная (хэш), «почти дубликаты» не ловятся.
- Список источников маленький (2 страницы Wikipedia и 1 PDF) и служит демонстрацией.
