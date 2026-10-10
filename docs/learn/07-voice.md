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
- Цепочка проверена вживую: аудио (синтезированный голос Windows) -> `faster-whisper` -> текст -> эндпоинт покрыт тестами с подменой.
- **Качество распознавания я не мерил.** На одном синтезированном роботизированном файле модель `base` дала
  неверный текст (`What the hell is Bina Fisha Lona-Shikmin?` вместо `What does beneficial ownership mean?`).
  Это один пример с искусственным голосом, а не оценка, но показывает риск: **неверная расшифровка = ответ на
  другой вопрос**. Модель `small` на том же файле тоже ошиблась (`What the S.B.F.I.E.S.H.E. LONER SHIP MIN`). Возможно, виноват сам синтезированный файл (роботизированный голос Windows, у меня нет живой записи), поэтому по этому тесту нельзя судить о качестве моделей: **перед демо запиши свой голос и проверь**.
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
- Качество не измерено (нет набора записей и WER); один тест с синтезированным голосом.
- `faster-whisper 1.2.1` несовместим с самой новой `av` (19.x): пришлось закрепить `av==16.1.0` (в `requirements-voice.txt`).
- Первый запрос медленный: загружается модель.
- Нет потоковой обработки и нет кнопки микрофона в Streamlit.
