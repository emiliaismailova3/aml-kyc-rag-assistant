# evals/: measurement scripts

Every number in the project README comes from one of these scripts or from `src/evaluate.py`
(RAGAS). Results are written to `evals/results/`. Scripts marked "LLM" make real calls to the
configured provider (Groq free tier by default: $0 billed, but daily token limits apply; most
scripts pause between requests to stay under Groq's 8000-tokens-per-minute limit).

| Script | What it measures | LLM? |
|---|---|---|
| `retrieval_hit_rate.py` | Document-level hit rate @1/@4/@8 on the 15 gold questions | no |
| `rerank_effect.py` | How the cross-encoder reranker changes the top chunks, and its CPU latency | no |
| `rerank_ab.py` | Refusals of base RAG at k=4 with the reranker off vs on; agent step-limit failures | LLM |
| `agent_routing_all.py` | Tool-selection accuracy of the agent on all 16 scenarios | LLM |
| `latency_cost.py` | p50/p95 latency, tokens and estimated cost per request type; end-to-end invoice → company matching; small-model routing saving | LLM |
| `forced_failover.py` | Primary provider down → fallback; all providers down → human escalation | LLM (1 call) |
| `security_eval.py` | Prompt-injection detection and false-positive rate; PII masking recall/precision | no |
| `sroie_eval.py` | Invoice pipeline on 30 real scanned receipts (ICDAR 2019 SROIE): field accuracy and whether invented values get accepted | LLM |

## Real receipts (SROIE)

Not committed (CC BY 4.0, ICDAR 2019; images are not redistributed here). Download the first 30
test receipts into `data/external/sroie/` with the Hugging Face datasets-server:

```python
import json, urllib.request, pathlib
url = "https://datasets-server.huggingface.co/rows?dataset=jsdnrs/ICDAR2019-SROIE&config=default&split=test&offset=0&length=30"
rows = json.load(urllib.request.urlopen(url))["rows"]
out = pathlib.Path("data/external/sroie"); out.mkdir(parents=True, exist_ok=True)
labels = []
for r in rows:
    row = r["row"]
    urllib.request.urlretrieve(row["image"]["src"], out / f"{row['key']}.jpg")
    labels.append({"file": f"{row['key']}.jpg", **row["entities"]})
(out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
```

## OCR language

Tesseract needs the Azerbaijani model (`aze.traineddata`) for `ə`, `ş`, `ğ`. If it lives outside
Tesseract's own folder, set `TESSDATA_PREFIX` (e.g. `C:/Users/you/tessdata`). Without it, OCR falls
back to English only and company-name accuracy on the synthetic invoices drops from 86.7% to 40%
(`evals/results/latency_cost_eng_only_ocr.json`).
