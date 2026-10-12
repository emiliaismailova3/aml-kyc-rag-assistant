# AI Document Assistant: cheat sheet (EN / RU)

## 1. The project in one sentence
EN: An AML/KYC assistant that answers compliance questions with cited sources, extracts and validates invoice data, matches companies to a reference list, and lets an agent use tools such as read-only SQL.
RU: Ассистент по AML/KYC: отвечает на вопросы со ссылками на источники, извлекает и проверяет данные счетов, сопоставляет компании со справочником, а агент пользуется инструментами, например SQL только на чтение.

## 2. Parts of the system
- **Knowledge base (RAG).** EN: PDFs are split into 700-character chunks (100 overlap), turned into vectors and stored; a question retrieves the 8 closest chunks and the LLM answers only from them, citing [n], or says "I don't know". RU: PDF режутся на куски по 700 символов (нахлёст 100), превращаются в векторы; вопрос находит 8 ближайших кусков, LLM отвечает только по ним со ссылками [n] или говорит «не знаю».
- **Invoice extraction.** EN: text layer or OCR → LLM returns JSON → validation rules → up to 2 repair attempts → otherwise human review. RU: текст или OCR → LLM возвращает JSON → проверка правилами → до 2 попыток исправления → иначе человеку.
- **Company matching.** EN: cascade VÖEN → normalized exact name → fuzzy (≥0.90) → embeddings (≥0.92); 0.75 and above goes to a human. RU: каскад VÖEN → точное имя после нормализации → похожее имя (≥0,90) → похожий смысл (≥0,92); от 0,75 решает человек.
- **Agent.** EN: decides which tool to call (knowledge base, calculator, web search, read-only SQL, invoice extraction), max 10 steps. RU: решает, какой инструмент вызвать (база знаний, калькулятор, интернет, SQL на чтение, извлечение счёта), не больше 10 шагов.
- **LLM reliability layer.** EN: one client for every LLM call: picks small or main model, retries transient errors with backoff + jitter, falls back to the next provider, escalates to a human, logs latency, tokens and cost (/stats). RU: один клиент для всех вызовов: выбирает малую или основную модель, повторяет временные ошибки с растущей паузой и разбросом, переключается на запасного провайдера, отдаёт человеку, пишет задержку, токены и стоимость (/stats).
- **Scheduled collection.** EN: Celery tasks fetch pages on a schedule (robots.txt, per-host rate limit), clean them, drop duplicates by hash, add new chunks. RU: задачи Celery по расписанию скачивают страницы (robots.txt, лимит на сайт), чистят, убирают дубли по хешу, добавляют новые куски.
- **Voice, streaming, translation, MCP.** EN: Whisper speech-to-text; answers streamed as server-sent events; translate and summarize endpoints with a term glossary; the tools exposed over MCP. RU: Whisper переводит речь в текст; ответ идёт потоком (SSE); перевод и резюме со словарём терминов; инструменты доступны по MCP.

## 3. Programs and libraries: what and why
- **Python** EN: main language. RU: основной язык.
- **FastAPI** EN: the HTTP API (/ask, /ask_agent, /ask_stream, /invoices/extract, /ask_voice, /translate, /summarize, /stats). RU: API-сервер со всеми эндпоинтами.
- **Streamlit** EN: demo user interface. RU: демо-интерфейс.
- **LangChain + LangGraph** EN: the tool-calling agent loop. RU: цикл агента с инструментами.
- **Chroma / PostgreSQL + pgvector** EN: vector storage; Chroma is simple and local, pgvector keeps vectors next to business data in one database (HNSW index, cosine). RU: хранение векторов; Chroma проще и локальная, pgvector держит векторы рядом с бизнес-данными в одной базе (индекс HNSW, косинус).
- **SQLAlchemy, psycopg** EN: database layer that works with SQLite and PostgreSQL. RU: слой работы с базой, SQLite и PostgreSQL.
- **sqlglot** EN: parses the SQL the LLM writes; only a plain SELECT on two views is allowed, LIMIT added, plus a read-only database role. RU: разбирает SQL от LLM; разрешён только SELECT по двум view, добавляется LIMIT, плюс роль только на чтение.
- **pdfplumber, Tesseract, OpenCV** EN: read the PDF text layer; OCR for scans (eng+aze); clean the image first (grayscale, deskew, Otsu threshold). RU: текст из PDF; OCR для сканов (eng+aze); подготовка картинки (серый цвет, выравнивание, порог Оцу).
- **Pydantic** EN: schema and business rules for the extracted invoice (VÖEN 10 digits, amount > 0, date not in the future, line items add up ±0.02). RU: схема и правила проверки счёта.
- **rapidfuzz** EN: fuzzy string similarity for company names. RU: нечёткое сравнение названий.
- **Celery + Redis** EN: background jobs and scheduler; Redis is the queue. RU: фоновые задачи и расписание; Redis это очередь.
- **BeautifulSoup** EN: HTML cleaning. RU: очистка HTML.
- **faster-whisper** EN: local speech-to-text. RU: локальное распознавание речи.
- **RAGAS** EN: answer quality metrics (faithfulness, answer relevancy, answer correctness) with an LLM as judge. RU: метрики качества ответов, судья это LLM.
- **MCP (mcp 2.x)** EN: standard protocol so any AI client can use our tools. RU: стандарт, чтобы любой ИИ-клиент мог пользоваться нашими инструментами.
- **Docker, GitHub Actions, pytest, ruff** EN: packaging, CI on every push, 249 tests, linting. RU: упаковка, CI на каждый push, 249 тестов, линтер.

## 4. Which model for which task
- **Answering questions (RAG).** EN: openai/gpt-oss-120b on Groq (main model), temperature 0. Why: strong enough to follow "answer only from the context" and cite; Groq is fast and has a free tier. RU: openai/gpt-oss-120b на Groq, температура 0. Почему: хорошо держит инструкцию «только по контексту» и ставит ссылки; Groq быстрый и бесплатный на старте.
- **Short simple prompts.** EN: the small model (gpt-oss-20b on Groq) when the prompt is ≤ 4000 characters (about 1000 tokens) and needs no tools. Why: cheaper and faster. RU: малая модель (gpt-oss-20b), если промпт до 4000 символов и без инструментов. Почему: дешевле и быстрее.
- **Agent with tools, retries, long prompts.** EN: always the main model. Why: tool calling and long context are where small models fail more. RU: всегда основная модель. Почему: на вызовах инструментов и длинном контексте малые модели ошибаются чаще.
- **Invoice extraction.** EN: JSON output checked by Pydantic; a repair attempt goes to the main model. Why: never trust the model's output, validate it. RU: ответ в JSON проверяется Pydantic; повторная попытка идёт на основную модель. Почему: модели не доверяем, проверяем.
- **Embeddings.** EN: BAAI/bge-small-en-v1.5, local, 384 dimensions. Why: free, private (no data leaves the machine), fast on CPU; the documents are in English. Option: OpenAI text-embedding-3-small. RU: bge-small-en-v1.5, локально, 384 измерения. Почему: бесплатно, данные не уходят наружу, быстро на CPU; документы на английском. Альтернатива: OpenAI text-embedding-3-small.
- **Fallback providers.** EN: chain configurable in .env (e.g. OpenAI gpt-4o / gpt-4o-mini, Anthropic Claude Sonnet / Haiku, Groq). Why: if one provider is down or rate-limited, the request still succeeds. RU: цепочка задаётся в .env. Почему: если провайдер лежит или упёрся в лимит, запрос всё равно выполнится.
- **RAGAS judge.** EN: a separate setting (EVAL_LLM_MODEL); in our run the judge was gpt-oss-120b, the same model that answers, so scores may be biased; numbers are a guide. RU: отдельная настройка; в нашем прогоне судья был той же моделью, что и отвечала, поэтому возможен перекос; цифры ориентир.
- **Speech-to-text.** EN: Whisper base, local on CPU (int8); option: OpenAI whisper-1. Tried small and a domain-word prompt: no consistent gain on 5 phrases. RU: Whisper base локально на CPU; вариант: OpenAI whisper-1. Пробовала small и подсказку терминов: стабильного выигрыша на 5 фразах нет.
- **OCR.** EN: Tesseract (eng+aze), free and local. RU: Tesseract (eng+aze), бесплатно и локально.

## 5. How I choose a model (general rules)
- EN: Start with the smallest model that passes the evaluation; move up only when the metric says so. RU: Начинаю с самой маленькой модели, которая проходит оценку; беру больше, только если метрика это показывает.
- EN: Check the capability the task needs: tool calling, JSON output, long context, the language (Azerbaijani). RU: Проверяю нужную способность: инструменты, JSON, длинный контекст, язык (азербайджанский).
- EN: Compare quality, latency and cost together; log all three. RU: Сравниваю качество, задержку и стоимость вместе; пишу все три в лог.
- EN: Sensitive data → prefer local models or a provider with the right data agreement. RU: Чувствительные данные → локальные модели или провайдер с нужным договором.
- EN: Limits matter: Groq's free tier caps a request at 8000 tokens, so the agent retrieves 4 chunks instead of 8. RU: Лимиты важны: Groq бесплатно ограничивает запрос 8000 токенов, поэтому агент берёт 4 куска, а не 8.
- EN: Temperature 0 for factual and structured tasks. RU: Температура 0 для фактов и структурированных задач.

## 6. Key lines for the interview
- EN: "I route short tool-free prompts to a small model and everything else to the main model, and every call goes through one client with retries, fallback and cost logging." RU: «Короткие промпты без инструментов идут на малую модель, остальное на основную; все вызовы через один клиент с повторами, запасным провайдером и логом стоимости».
- EN: "I don't trust model output: the invoice JSON is validated by rules, and anything that fails twice goes to a human." RU: «Модели не доверяю: JSON счёта проверяется правилами, а то, что не прошло дважды, идёт человеку».
- EN: "I measure before I claim: RAGAS for answers, accuracy for invoices, precision and recall for matching, WER for speech." RU: «Сначала измеряю, потом утверждаю: RAGAS для ответов, точность для счетов, precision и recall для сопоставления, WER для речи».
