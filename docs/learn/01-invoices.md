# 01. Извлечение данных из инвойсов (`src/invoices/`)

## Что это делает (простыми словами)
Ты отдаёшь системе файл с инвойсом (PDF или фото). Она возвращает аккуратную таблицу полей:
кто продавец, его VÖEN, номер инвойса, дата, сумма, валюта. А если данным нельзя верить,
она честно говорит «покажите человеку» (`needs_human_review`).

**Аналогия.** Представь стажёра-бухгалтера. Он смотрит на бумагу (OCR), переписывает данные
в форму (LLM), а старший бухгалтер проверяет форму по правилам («сумма больше нуля»,
«позиции дают в сумме итог»). Если ошибка, форма возвращается стажёру с пометкой, что
неверно. После двух неудачных попыток она уходит человеку.

## Схема потока
```
файл (PDF / картинка)
   |
   v
есть текстовый слой? --да--> pdfplumber читает текст (быстро, точно)
   |нет (скан/фото)
   v
серый -> выпрямить наклон -> чёрно-белый (порог Оцу) -> Tesseract (eng+aze)
   |
   v
текст --> LLM --> JSON --> Pydantic проверяет поля и бизнес-правила
                              |ошибка?                 |ок
                              v                        v
          текст ошибки возвращается в LLM        InvoiceData (готово)
          (максимум 2 повтора)
                              |всё ещё ошибка
                              v
                    needs_human_review = true
```

## Файл за файлом
- `schema.py`: формы данных и правила. `InvoiceData` (поля), `LineItem` (строка таблицы).
  Валидаторы: VÖEN ровно 10 цифр, сумма > 0, дата не из будущего, сумма позиций = итог
  (допуск 0.02 на округление).
- `ocr.py`: `extract_text(path)` возвращает `(текст, метод)`. Если у PDF есть текстовый слой
  (≥30 символов), читаем его. Иначе рендерим страницу в 300 DPI и вызываем `ocr_image`.
  `preprocess` делает серый цвет, `_deskew` (выпрямляет наклон по `minAreaRect`) и порог Оцу.
  `_ocr_languages` берёт `eng+aze`, а если азербайджанской модели нет, то только `eng`.
- `extract.py`: `extract_from_text` говорит модели «верни один JSON», парсит его
  (`parse_json_reply` убирает ```-обёртки), валидирует. При ошибке добавляет в диалог
  ответ модели и сообщение «JSON неверен: ...». `extract_invoice` — вся цепочка для файла.
- `evaluate.py`: прогоняет все 15 синтетических инвойсов, сравнивает 6 полей с эталоном
  (`labels.json`), считает точность по каждому полю и типу файла. Файлы, которым нужен OCR,
  а Tesseract не установлен, **пропускаются и считаются отдельно**. Цифра не выдумывается.
- `scripts/generate_synthetic_invoices.py`: делает 15 инвойсов: 5 текстовых PDF,
  5 чистых PNG, 3 «плохих скана» JPG и 2 PDF-скана.
- Эндпоинт `POST /invoices/extract` в `src/api.py` принимает файл и отдаёт `ExtractionResult`.

## Вопросы на собеседовании (EN)
1. **Why not just send the PDF to a vision LLM?** Text-layer PDFs are read exactly and for free.
   OCR/vision is only the fallback, and I keep documents on my own machine.
2. **How do you make sure the output is valid?** Pydantic schema plus business rules; failures are
   fed back to the model, max two retries, then a human reviews it.
3. **What if the LLM hallucinates a number?** The line-item sum must equal the total and the VOEN
   must be 10 digits. Those rules catch many hallucinations, but not all (see limitations).
4. **Why Decimal for money?** Floats make rounding errors (0.1 + 0.2), which is unacceptable in finance.
5. **What does deskew do?** Scans are slightly rotated. Straightening them lifts OCR accuracy.
6. **Why Otsu thresholding?** It picks the black/white cut-off automatically from the histogram.
7. **How do you measure quality?** A hand-labelled synthetic set; per-field accuracy per file kind.
8. **How is it tested without API keys?** The LLM is a scripted fake. The text-layer PDFs are real.
9. **How would you scale it?** Put extraction in a Celery queue, store results in Postgres, and
   send low-confidence cases to a review UI.
10. **Why `needs_human_review` instead of best guess?** In finance a confident wrong value is worse than a delay.

## Результат (честно)
Оценка на всех 15 синтетических инвойсах с настоящим Tesseract 5.4 (`eng+aze`):
- VÖEN, номер, дата, сумма, валюта: **100%**;
- название компании: **86.7%** (13 из 15). Две «ошибки» это потеря точек над буквами: «Ticarət» стало «Ticaret»,
  «Kəpəz» стало «Kepez». Модуль сопоставления (02) всё равно находит правильную компанию по обоим названиям.
- Группы: PDF с текстовым слоем 5/5, чистые PNG 3/5 по названию, «шумные» JPG и PDF-сканы 5/5.

## Ограничения (говорить честно)
- Все данные синтетические и сделаны по **одному шаблону** чистым шрифтом. Шум мягкий, поэтому 100% на «шумных»
  файлах не говорит о качестве на реальных фото, печатях, рукописи и разных макетах.
- Tesseract сам по себе ошибается на `ə` и других азербайджанских буквах, даже с моделью `aze`. Поэтому сопоставление
  компаний нормализует названия и не зависит от точек над буквами.
- Проверки ловят несогласованность, но если модель уверенно перепутает покупателя с продавцом, правила это не поймают.
- На Windows установщик Tesseract не добавляет его в PATH: код ищет `TESSERACT_CMD` или `C:/Program Files/Tesseract-OCR`.
