# 07. Голосовые вопросы (`src/voice.py`, `POST /ask_voice`)

## Что это делает
Ты отправляешь аудиофайл с вопросом. Whisper превращает речь в текст, и этот текст идёт к тому же агенту,
что и обычный вопрос. В ответе ты получаешь **расшифровку** (чтобы видеть, как тебя поняли) и ответ агента.

**Аналогия.** Секретарь, который сначала печатает то, что ты сказал, а потом передаёт записку эксперту.
Если секретарь неправильно расслышал, эксперт ответит на другой вопрос. Поэтому расшифровка возвращается
пользователю.

## Схема
```
аудио (wav/mp3/m4a/ogg/webm/flac) --> проверка типа и размера (<= 25 МБ)
   |
   v
Whisper: WHISPER_PROVIDER=local (faster-whisper, CPU, бесплатно) или openai (облачный API)
   |
   v
текст пустой? --да--> 422 «речь не распознана»
   |нет
   v
агент (те же 5 инструментов) --> {transcript, answer, sources, tool_calls}
```

## Файл за файлом
- `src/voice.py`:
  - `transcribe(audio, filename, language)` выбирает бэкенд по `WHISPER_PROVIDER`.
  - `_transcribe_local`: пишет байты во временный файл (декодеру нужен путь) и вызывает `faster-whisper`
    (`compute_type="int8"`, `vad_filter=True` вырезает паузы и шум). Модель загружается лениво и один раз.
  - `_transcribe_openai`: вызов `audio.transcriptions.create(model="whisper-1")`, нужен `OPENAI_API_KEY`.
  - `VoiceUnavailable`: понятная ошибка, если пакет или ключ не настроены -> API отдаёт 503.
- `src/api.py`: `POST /ask_voice` (необязательное поле `language`, например `en` или `az`). Блокирующие вызовы
  идут через `run_in_threadpool`, чтобы не останавливать сервер.
- `requirements-voice.txt`: `faster-whisper` вынесен отдельно, потому что тянет тяжёлые зависимости и модель.

## Как попробовать
```bash
pip install -r requirements-voice.txt
curl -F "file=@question.wav" -F "language=en" http://localhost:8000/ask_voice
```

## Результат (честно)
Проверено на двух реальных записях автора (английский, аудио из WhatsApp), через `POST /ask_voice`:
- «What is 450 times 37?»: расшифровано точно, агент вызвал калькулятор, ответ **16,650**.
- «What does beneficial ownership mean?»: обе модели (`base` и `small`) услышали **«beneficial over-ship»**.
  Но агент всё равно нашёл правильный документ и ответил верно (в ответе он сам отметил странное слово). Так получилось
  потому, что поиск по смыслу устойчив к одному неверному слову, а не потому, что распознавание хорошее.
- Две записи: это проверка «работает ли цепочка», а **не измерение точности** (нет набора записей и WER).
  Раньше на синтезированном голосе Windows обе модели ошиблись полностью, поэтому качество зависит от записи.
- Азербайджанский язык поддерживается Whisper, но я его не проверял.

## Вопросы на собеседовании (EN)
1. **Why return the transcript?** Speech recognition can be wrong; the user must see what the system understood.
2. **Local vs API Whisper?** Local is free and private but slower and needs a model download; the API is fast but sends audio out.
3. **What is `int8` compute?** Quantized weights: less memory and faster on CPU, with a small accuracy cost.
4. **What does `vad_filter` do?** Voice activity detection removes silence/noise so Whisper hallucinates less.
5. **How would you evaluate it?** Word error rate (WER) on a labelled set of recordings in the target languages.
6. **What if the audio is empty?** The API returns 422 instead of sending an empty question to the agent.
7. **Why a thread pool?** Transcription and the agent are blocking calls; running them in the event loop would freeze the server.
8. **How do you limit abuse?** Allowed extensions and a 25 MB size cap.
9. **Streaming?** Not implemented; the whole file is uploaded first.
10. **Privacy?** Audio is processed in memory/a temp file that is deleted; with the OpenAI option it leaves your server.

## Ограничения
- Качество не измерено (нет набора записей и WER); проверено на двух записях и одном синтезированном файле.
- `faster-whisper 1.2.1` несовместим с самой новой `av` (19.x): пришлось закрепить `av==16.1.0` (в `requirements-voice.txt`).
- Первый запрос медленный: загружается модель.
- Нет потоковой обработки и нет кнопки микрофона в Streamlit.
