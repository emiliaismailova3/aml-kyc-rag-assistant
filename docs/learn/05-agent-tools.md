# 05. Новые инструменты агента (`sql_query`, `extract_invoice`)

## Что это делает
Агент теперь умеет пять вещей: искать в базе документов, считать, искать в интернете и **две новых**:
- `sql_query`: отвечает на вопросы про данные в базе («сколько инвойсов в EUR и какая сумма?»);
- `extract_invoice`: читает файл инвойса по имени и возвращает поля + найденную компанию из справочника.

**Аналогия.** Агент — менеджер с пятью помощниками. Вопрос про закон уйдёт в библиотеку (поиск по документам),
про арифметику — к калькулятору, про цифры в бухгалтерии — к аналитику, который **может только читать две
витрины данных** и не может ничего менять, а про конкретный файл — к оператору, который сканирует документ.

## Схема безопасности SQL (несколько слоёв)
```
SQL от модели
  |-> 1. разобрать sqlglot: ровно один SELECT? (нет DROP/INSERT/;-цепочек/UNION/WITH/INTO)
  |-> 2. все таблицы только из {v_invoices, v_companies}? (base-таблицы и pg_* запрещены)
  |-> 3. LIMIT обязателен (нет -> добавим 50; больше 100 -> отказ)
  |-> 4. нет опасных функций (pg_sleep, pg_read_file, set_config ...)
  |-> 5. выполнить в READ ONLY транзакции, таймаут 3 с, под ролью assistant_ro (видит только 2 витрины)
```
Любой слой по отдельности можно было бы обойти ошибкой, поэтому их несколько.

## Файл за файлом
- `src/sql_tool.py`: `validate_sql` (слои 1-4), `run_readonly_query` (слой 5), `format_rows` (JSON, обрезка).
- `src/db.py`: витрины `v_invoices` (без VÖEN!) и `v_companies`; выдача прав роли `assistant_ro`.
- `src/agent.py`: инструменты `sql_query` и `extract_invoice` (ищет файл только по имени в `data/invoices`,
  `data/uploads`: путь вида `../../.env` отбрасывается), список `TOOLS`, обновлённый системный промпт.
- `scripts/seed_demo_db.py`: заполняет БД демо-данными (15 инвойсов из эталонных меток, без LLM).
- `data/agent_test_scenarios.json`: сценарии t11-t16 (3 на SQL, 2 на инвойс, 1 «похоже на данные, но это вопрос про закон»).
- `scripts/check_agent_routing.py`: запускает агента на сценариях и проверяет, какой инструмент он реально вызвал.

## Результат (честно)
Живой прогон на 6 новых сценариях: **6/6** выбрали верный инструмент, и ответы совпали с эталоном
(4 инвойса EUR на 38 530.64; крупнейший 27 560.73 EUR у "Aygün Ticarət"; 0 на проверке; поля двух инвойсов верные).
Это маленькая выборка и один прогон. Старые сценарии t01-t10 заново не прогонялись.

## Вопросы на собеседовании (EN)
1. **How do you stop an LLM from running destructive SQL?** Layers: parser allows one SELECT, a table whitelist of
   views, a mandatory LIMIT, blocked functions, a read-only transaction with a timeout, and a DB role that can
   only read those views.
2. **Why views, not tables?** Views expose only needed columns (no VOEN), and permissions are granted on the view.
3. **What about prompt injection?** A malicious document could tell the model to dump data. The permissions are
   enforced outside the model, so the worst case is a read of the two views.
4. **Why sqlglot instead of regex?** Regex misses things like `FROM a, b`, quoting or schema-qualified names.
5. **Why is a CTE rejected?** The CTE name would look like an unknown table; simplicity over coverage. It is a limitation.
6. **How does the agent pick a tool?** Function calling: it reads each tool's description (docstring) and arguments.
7. **How do you test routing?** Scenario file with the expected tool, a script that records the tools actually called.
8. **What if the model hallucinates a file name?** The tool answers "not found" with the list of real files.
9. **Why not let the agent write to the DB?** Principle of least privilege; writes should go through validated code paths.
10. **What bug did the live run find?** SQLite's read-only flag leaked into a shared connection and broke log writes;
    fixed with a separate engine and a regression test.

## Ограничения
- Агент берёт из базы знаний 4 фрагмента (не 8) из-за лимита запроса Groq; пять инструментов тоже занимают токены.
- `extract_invoice` для сканов требует Tesseract (на этой машине его нет): проверен только на PDF с текстовым слоем.
- RAGAS-оценка агента (3 инструмента) делалась на ветке `main`; на этой ветке агент другой, оценку нужно пересчитать.
- Выборка сценариев маленькая; повторяемость между запусками не измерялась.
