# 04. Надёжность LLM (`src/llm_client.py`, `src/llm_stats.py`, `GET /stats`)

## Что это делает
Все обращения к языковой модели идут через одну обёртку. Она:
1. выбирает **маленькую** или **большую** модель (дёшево против качественно),
2. при временной ошибке (лимит, 5xx, таймаут) **ждёт и повторяет**,
3. если провайдер лежит, **переключается на следующего** (например OpenAI -> Anthropic -> Groq),
4. если никто не отвечает, говорит «**передайте человеку**» (`escalate_to_human`),
5. **записывает каждую попытку**: модель, время, токены, стоимость, статус (без текста вопросов и ответов).

**Аналогия.** Диспетчер такси. Короткий заказ отдаёт дешёвой машине, длинный дальний рейс — большой.
Если машина не берёт трубку, перезванивает через 1, 2, 4 секунды (с небольшим случайным разбросом, чтобы
все диспетчеры не звонили одновременно). Если парк не отвечает, звонит в другой парк. Если не отвечает
никто, зовёт менеджера. Всё записывает в журнал.

## Схема
```
complete(messages)
   |
   v
choose_tier: короткий и без инструментов? -> small, иначе main
   |
   v
для каждого провайдера в цепочке:
    попытка -> успех? --да--> записать в llm_calls, вернуть ответ
       |ошибка
       v
    временная (429/5xx/таймаут)? -нет-> следующий провайдер
       |да, и попыток < 4
       v
    пауза 1с, 2с, 4с (+ jitter) -> новая попытка
   |
   v
все провайдеры упали -> LLMUnavailable (needs_human) -> API: 503/429 + escalate_to_human=true
```

## Файл за файлом
- `config.py`: `get_provider_chain()` читает `LLM_FALLBACK_CHAIN` и пропускает провайдеров без ключа.
  `DEFAULT_PRICES` — приблизительные цены за 1М токенов (для оценки, переопределяются через `LLM_PRICES_JSON`).
- `llm_client.py`:
  - `choose_tier`: простое объяснимое правило (≤4000 символов, без инструментов, не повтор -> small).
  - `is_retryable`: повторяем только временные ошибки; 400/401/404 сами не пройдут.
  - `backoff_delay`: экспонента 1,2,4,... до 30 с, случайно уменьшенная до 50-100% (jitter).
  - `OpenAICompatibleProvider` / `AnthropicProvider`: тонкие адаптеры над SDK (`max_retries=0`, чтобы повторы
    делали мы и могли их записать).
  - `LLMClient.complete`: основной цикл (схема выше). Ошибка записи в журнал никогда не ломает запрос.
- `llm_stats.py`: считает p50/p95 задержки (по успешным вызовам), долю ошибок, стоимость, токены, разбивку по моделям.
- `llm_callbacks.py`: агент работает через LangChain, поэтому его вызовы пишет колбэк `LLMCallLogger`.
- `api.py`: обработчик `LLMUnavailable` и `GET /stats`.
- Подключено в: `rag.py` (ответ RAG), `invoices/extract.py` (повторы идут на большую модель).

## Вопросы на собеседовании (EN)
1. **Which errors do you retry?** Only transient ones: 429, 5xx, 408/409, timeouts, dropped connections.
2. **Why jitter?** Without it, many clients that failed together retry together and overload the server again.
3. **Why disable the SDK's own retries?** I need to log each attempt and let a failure trigger the fallback.
4. **How does routing work?** Short, tool-free first attempts go to the small model; long prompts, tool use and
   repair retries go to the main model. It is a rule, not a trained router, so it is easy to explain.
5. **How do you estimate cost?** Tokens x price per 1M from a table; the table is approximate and configurable.
6. **What do you log, and what not?** Model, latency, tokens, cost, status, error type. Never prompts or answers
   (they may contain personal or financial data).
7. **What happens if all providers fail?** An exception flagged `needs_human`; the API returns 503/429 with
   `escalate_to_human: true`; invoice extraction returns `needs_human_review`.
8. **Why p95, not the average?** The average hides slow outliers; users feel the tail.
9. **How would you add a circuit breaker?** Skip a provider for N seconds after M consecutive failures.
   (Not implemented; a limitation.)
10. **Is the agent covered?** Only partly: logged by a callback and uses the SDK's retries, but no fallback chain.

## Ограничения (честно)
- Фолбэк на Anthropic **проверен только на моках**: у меня нет ключа Anthropic. Живой вызов проверен для Groq.
- У агента нет цепочки провайдеров (LangChain tool-calling привязан к одному клиенту).
- Цены приблизительные; без «автоматического выключателя» (circuit breaker); задержки считаются на уровне попытки.
