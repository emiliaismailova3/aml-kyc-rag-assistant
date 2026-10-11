"""Measure speech recognition on your own recordings: word error rate (WER) per phrase.

1. Record each phrase from data/voice_eval/phrases.json and save the audio next to it under the
   listed file name (data/voice_eval/*.wav|mp3|m4a|mp4|ogg|webm). The recordings are git-ignored.
2. Run:  python -m scripts.check_voice
   Missing recordings are skipped and reported. Results go to data/eval_results/voice_wer.json.

WER = (substitutions + deletions + insertions) / words in the reference. 0.0 is perfect; it can
exceed 1.0. Text is lower-cased and stripped of punctuation first, and Azerbaijani letters are
compared exactly (ə is not e), because a wrong letter is a different word.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHRASES = ROOT / "data" / "voice_eval" / "phrases.json"
OUTPUT = ROOT / "data" / "eval_results" / "voice_wer.json"


def normalize(text: str) -> list[str]:
    text = text.replace("İ", "i").replace("I", "ı").lower()  # Turkic casing: I -> ı, İ -> i
    return re.sub(r"[^\w\s]", " ", text).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("The reference phrase is empty.")
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def main() -> int:
    from src.voice import transcribe

    phrases = json.loads(PHRASES.read_text(encoding="utf-8"))
    rows, missing = [], []
    for item in phrases:
        audio_path = PHRASES.parent / item["file"]
        if not audio_path.exists():
            missing.append(item["file"])
            continue
        started = time.perf_counter()
        heard = transcribe(audio_path.read_bytes(), audio_path.name, language=item["language"])
        rows.append({
            "file": item["file"], "language": item["language"], "reference": item["text"], "heard": heard,
            "wer": round(word_error_rate(item["text"], heard), 3),
            "seconds": round(time.perf_counter() - started, 1),
        })
        print(f"[{item['language']}] WER {rows[-1]['wer']:.2f}  said: {item['text']}\n           heard: {heard}")

    report = {"phrases_scored": len(rows), "missing_recordings": missing, "rows": rows}
    for language in sorted({r["language"] for r in rows}):
        values = [r["wer"] for r in rows if r["language"] == language]
        report[f"mean_wer_{language}"] = round(statistics.mean(values), 3)
        print(f"mean WER ({language}): {report[f'mean_wer_{language}']}  over {len(values)} phrase(s)")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if missing:
        print(f"\nNo recording yet for: {', '.join(missing)}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
