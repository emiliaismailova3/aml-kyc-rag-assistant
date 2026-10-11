# 03. PostgreSQL + pgvector (`src/db.py`, `src/pgvector_store.py`)

## Что это делает
Раньше фрагменты документов лежали в локальной папке Chroma. Теперь их можно хранить в настоящей
базе данных PostgreSQL с расширением **pgvector** (`VECTOR_BACKEND=pgvector`). Там же лежат таблицы
компаний, инвойсов и логов. Chroma остаётся вариантом по умолчанию, ничего не сломано.

**Аналогия.** Chroma — это записная книжка у тебя на столе. PostgreSQL — это корпоративный архив с
каталогом: много людей, права доступа, резервные копии. pgvector — это особый каталог в архиве, где
карточки ищутся «по смыслу», а не по алфавиту.

## Схема
```
PDF -> фрагменты -> эмбеддинги (384 числа) -> таблица chunks (колонка vector(384))
                                                  |
вопрос -> эмбеддинг -> ORDER BY embedding <=> вопрос LIMIT k   (HNSW-индекс ускоряет поиск)
                                                  |
                              Document(source, page, chunk_id) -> те же цитаты [n]
```

## Файл за файлом
- `src/config.py`: `VECTOR_BACKEND` (chroma|pgvector), `get_database_url()`. Если `DATABASE_URL`
  пуст, используется локальный SQLite `logs/app.db`, поэтому всё работает без Postgres.
- `src/db.py`: таблицы через SQLAlchemy Core: `companies`, `invoices`, `request_logs`, `llm_calls`,
  `collected_documents`, и два представления `v_invoices` / `v_companies` (без VÖEN, без лишних
  колонок) для SQL-инструмента агента. `init_db()` создаёт всё (повторный запуск безопасен) и выдаёт
  роли `assistant_ro` право SELECT только на представления.
- `src/pgvector_store.py`: класс `PGVectorStore` с тем же методом `similarity_search(query, k)`, что у
  Chroma. `add_documents` вставляет пачками по 200, `ON CONFLICT DO NOTHING` делает загрузку повторяемой.
  Вектор передаётся строкой `'[0.1,0.2,...]'` и приводится `CAST(... AS vector)`. Оператор `<=>` — это
  косинусное расстояние. Имя таблицы проверяется регулярным выражением (его нельзя передать параметром).
- `src/vectorstore.py`: `build_vectorstore` и `load_vectorstore` выбирают бэкенд по `VECTOR_BACKEND`.
- `docker-compose.yml`: сервисы `postgres` (образ `pgvector/pgvector:pg16`) и `redis` в профиле `full`.
  `docker/postgres/init.sh` создаёт расширение и роль `assistant_ro`.

## Как запустить
```bash
docker compose --profile full up -d postgres redis
export DATABASE_URL=postgresql://assistant:assistant@localhost:5432/assistant VECTOR_BACKEND=pgvector
python -m src.vectorstore        # загрузит 2176 фрагментов (около 6 минут на CPU)
```

## Вопросы на собеседовании (EN)
1. **Why pgvector instead of a dedicated vector DB?** One database for vectors and business data
   (joins, transactions, backups, access control); fine for this scale.
2. **What is HNSW?** An approximate nearest-neighbour graph index: very fast search for a small recall loss.
3. **What does `<=>` mean?** Cosine distance between two vectors (smaller = more similar).
4. **Why `vector_cosine_ops`?** The index must use the same distance as the query, and the embeddings are normalized.
5. **How is ingestion idempotent?** A unique `chunk_id` and `ON CONFLICT DO NOTHING`.
6. **Why SQLAlchemy Core?** The same table definitions work on SQLite (tests, zero setup) and PostgreSQL.
7. **Why a read-only role and views?** The agent's SQL tool can only read two views, with no sensitive columns.
8. **Chroma vs pgvector, when each?** Chroma for a local demo; pgvector when you need shared, durable, secured storage.
9. **How would you tune recall?** `hnsw.ef_search`, index parameters `m` / `ef_construction`.
10. **What about embedding-dimension changes?** The column is `vector(384)`; a different model needs a new column/table and re-embedding.

## Ограничения
- Эмбеддинги считаются на CPU: загрузка всего корпуса занимает минуты.
- Размерность вектора фиксирована (384), смена модели требует переиндексации.
- Миграций схемы нет (`create_all` создаёт только недостающее); для продакшена нужен Alembic.
- Сравнение качества Chroma и pgvector по RAGAS я не делал; проверил только, что поиск возвращает осмысленные
  фрагменты с теми же метаданными для цитат.
