# 02. Сопоставление компаний (`src/matching/`)

## Что это делает
На инвойсе написано `"XEZER LOGISTIKA" LLC`. В базе компания лежит как `Xəzər Logistika MMC`.
Модуль находит, что это одна и та же фирма, и возвращает её id, уверенность (score) и способ (method).
Если не уверен, он не угадывает, а просит человека (`needs_human_review`).

**Аналогия.** Администратор на ресепшене ищет гостя в списке. Сначала по паспорту (VÖEN): это точно.
Потом по имени, не обращая внимания на заглавные буквы и точки над буквами. Потом «а, наверное,
опечатка» (fuzzy). Потом «по смыслу похоже» (embedding). Если всё ещё сомневается, зовёт менеджера.

## Схема
```
имя (+ VÖEN)
   |
   v
1. VÖEN есть в таблице? ------------------ да -> совпадение, score 1.0, method=voen
   |нет
2. нормализованное имя совпало точно? ----- да -> score 1.0, method=exact_name
   |нет
3. rapidfuzz (опечатки) >= 0.90? ---------- да -> method=fuzzy
   |нет
4. embedding-сходство >= 0.92? ------------ да -> method=embedding
   |нет
5. лучший score >= 0.75? ------------------ да -> кандидат, но needs_human_review=true
   |нет
   v
 совпадения нет
```

## Файл за файлом
- `normalize.py`: `normalize_name` приводит имя к сравнимому виду. Сначала заменяет `İ` и `I` на `i`
  (иначе Python оставляет «точку» от `İ`), потом нижний регистр, потом `ə→e ğ→g ı→i ö→o ü→u ş→s ç→c`,
  убирает кавычки и знаки, затем удаляет юридические формы (MMC, QSC, ASC, LLC...).
- `matcher.py`: `CompanyMatcher.match(name, voen)` проходит по шагам схемы. Векторы названий считаются
  лениво (один раз) и сравниваются через скалярное произведение нормированных векторов (косинус).
  Функцию эмбеддинга можно подменить, поэтому тесты не грузят модель.
- `evaluate.py`: прогоняет 52 размеченные пары (`data/reference/matching_pairs.json`) и считает
  precision и recall. Для «принято автоматически» нужно: найдена компания и нет флага ручной проверки.
- `scripts/generate_reference_data.py`: создаёт 50 вымышленных компаний и пары с искажениями
  (регистр, без диакритики, без юрформы, опечатки, перестановка слов, и «почти похожие», которых в
  таблице нет).

## Результат (честно)
52 пары: precision **1.0**, recall **1.0**, 3 «почти похожих» отрицательных примера ушли на ручную проверку.
Но: набор синтетический и небольшой, а пороги (0.90 / 0.92 / 0.75) я выбрал заранее, не подгоняя
под него. На реальных данных цифры будут хуже. Первая версия набора давала «идеальные» 100%, потому что
была слишком лёгкой (почти всё решала нормализация), поэтому я добавил сложные случаи.

## Вопросы на собеседовании (EN)
1. **Why a cascade and not just embeddings?** Cheap, exact steps first; they are explainable and
   never wrong when they fire. Embeddings are slower and fuzzy, so they come last.
2. **Why strip legal forms?** "X MMC" and "X LLC" are the same business; the form adds noise.
3. **What is the Turkish dotted/dotless I problem?** Python's `lower()` turns `İ` into `i` + a combining
   dot, so "İPƏK" would not equal "ipək". I replace it before lower-casing.
4. **Precision vs recall here?** Precision: automatic matches that are right. Recall: real matches found
   automatically. In finance I prefer high precision and send doubtful cases to a human.
5. **How did you pick thresholds?** Checked on the labeled pairs; they were set once, not tuned on the result.
   With more data I would plot a precision/recall curve.
6. **Why `token_sort_ratio`?** It ignores word order but still penalizes different words.
7. **Why a VOEN match is certain?** It is a legal tax identifier; names are not unique.
8. **What if two companies have similar names?** The score is lower than the threshold, so a human decides.
9. **Is the embedding model good for Azerbaijani?** It is an English model (bge-small); after
   normalization to Latin letters it works, but a multilingual model would be better.
10. **How would you scale to 1M companies?** Store vectors in pgvector with an HNSW index and fetch top-k
    candidates instead of comparing against everything.

## Ограничения
- Данные синтетические и маленькие (52 пары); реальные названия грязнее.
- Эмбеддинг-модель английская.
- Сравнение идёт со всеми компаниями подряд, что нормально для 50, но не для миллионов.
