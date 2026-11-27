# Roadmap: from compliance RAG to an Azerbaijani voice banking assistant

Goal: a voice and text assistant for retail bank customers in Azerbaijan. It answers in Azerbaijani or
Russian from the **public** documents of Azerbaijani banks (tariffs, card and deposit conditions, FAQ) and the
Central Bank's rules, can compare banks, and hands anything it cannot handle safely to a human operator.
The full brief is written in stage 1 (`docs/PRODUCT.md`).

## Data principles (where the trust comes from)

- **Real documents**: public bank and CBAR documents, every source listed with a link. The project uses them
  as a knowledge source and never presents itself as any bank (no bank names or logos as branding;
  a "not affiliated" note in the README).
- **Real questions**: the eval set comes from questions customers actually ask (bank FAQ pages, public
  comments, forums), paraphrased, with no names or personal data kept.
- **Real voices**: recordings from 10–15 people of different ages (with their consent) in az, ru and mixed,
  plus an open speech set (FLEURS).
- **Real users**: the demo is tested with 5–10 people, ideally contact-centre or bank staff, and the results
  are published.
- **Synthetic only where data is private** (customers, application status), and labelled as synthetic.

One stage per week (roughly). Each stage follows the same loop:

1. **Learn** the concepts listed for the stage.
2. **Read** the code of this stage and explain it in your own words (a short note in `docs/notes/`).
3. **Change one thing and measure it**: write the number before and after into `docs/EVALUATION.md`.

Rule: no code goes in that its author cannot explain line by line.

## Phase 1: text core in Azerbaijani and Russian (weeks 1–6)

### Stage 1. Problem framing
- Concepts: user, use case, scope, business metric vs. model metric, baseline.
- Do: write `docs/PRODUCT.md`: user, problem, scope and non-goals, metrics, risks; check 20–30 Baku AI/ML vacancies and note how many
  ask for voice / speech vs. RAG / LLM / agents.
- Done when: the brief is written and only targets that need a baseline are left open.

### Stage 2. Data
- Concepts: document parsing, OCR, cleaning, chunking, chunk size and overlap, metadata.
- Do: collect public documents of 3–5 banks (tariffs, cards, deposits, loans, FAQ) in az and ru, with a
  source list; collect 100+ real customer questions; ingest the documents next to the CBAR ones; try two
  chunk sizes.
- Done when: the new documents are indexed with source and page, and the chunk-size result is recorded.

### Stage 3. Retrieval
- Concepts: embeddings, cosine similarity, vector index (HNSW), BM25, hybrid search, reranking, top-k,
  recall@k, MRR.
- Do: replace the English-only embedding model with a multilingual one; add BM25 + vector hybrid search.
- Done when: recall@k on az / ru / mixed questions is measured for old vs. new.

### Stage 4. Generation
- Concepts: system prompt, grounding, citations, temperature, structured output, hallucination.
- Do: answers in the customer's language, short enough to be spoken; never invent fees or limits.
- Done when: every money-related question in the eval set is answered correctly or refused.

### Stage 5. Evaluation
- Concepts: golden set, LLM-as-a-judge, faithfulness, answer correctness, regression test, CI gate.
- Do: 60–100 questions (az, ru, mixed); run the evaluation in CI on a small subset and fail the build
  when quality drops.
- Done when: a pull request that breaks retrieval turns CI red.

### Stage 6. Agent and tools
- Concepts: function calling, ReAct, LangGraph state, tool limits, escalation, MCP.
- Do: tools for currency rates, application status and hand-off to an operator; score the agent on the eval set.
- Done when: tool choice and answer quality of the agent are measured against plain RAG.

## Phase 2: voice (weeks 7–10)

### Stage 7. Speech recognition (ASR)
- Concepts: ASR / STT, WER, CER, VAD, sampling rate, code-switching.
- Do: record 10–15 people (with consent) reading real customer questions, including numbers and mixed
  az/ru phrases; measure several ASR models on these recordings and on FLEURS.
- Done when: a WER / CER table picks the model, with latency next to accuracy.

### Stage 8. Speech synthesis (TTS) and text normalization
- Concepts: TTS, voices, text normalization (numbers, dates, currency, card numbers → words), SSML.
- Do: compare available Azerbaijani TTS options; write the normalizer with tests.
- Done when: amounts like "1 250,50 AZN" are spoken correctly in az and ru.

### Stage 9. Real-time pipeline
- Concepts: streaming, WebSocket, time to first audio, latency budget, barge-in, turn-taking.
- Do: stream every step (ASR → agent → TTS); let the customer interrupt the answer.
- Done when: p50 time to first audio is measured and at or under the target in `docs/PRODUCT.md`.

### Stage 10. Safety for a customer-facing assistant
- Concepts: PII in audio and transcripts, prompt injection by voice, guardrails, least privilege, escalation.
- Do: mask PII in transcripts and logs; test spoken injection attempts; safe refusals.
- Done when: the security eval covers voice inputs.

## Phase 3: product (weeks 11–12)

### Stage 11. Serving and operations
- Concepts: auth, rate limiting, caching, tracing, cost per conversation, monitoring, feedback loop.
- Do: auth + rate limit on the API; live tracing; thumbs up / down stored and turned into new eval cases;
  a user test with 5–10 people (ideally contact-centre or bank staff), results written up.

### Stage 12. Deployment and presentation
- Concepts: CI/CD, secrets, cloud deployment.
- Do: public demo with a request limit; a 2-minute video; README rewritten around the voice assistant;
  repository renamed to match the product.
