# Engineering notes

Debugging notes from running the project against a real Groq key and the RAGAS
judge. Moved out of the README to keep it short; nothing here is needed to run
the project.

Running this against a real Groq key (rather than just importing against mocks)
surfaced a few issues that are worth noting for anyone hitting the same thing:

- **Decommissioned default model.** The original default, `llama-3.3-70b-versatile`,
  returned `404 model_not_found` on Groq. Provider-hosted "serverless" model
  catalogs change over time independently of this repo's code — if a model
  disappears, check the provider's current list and update `LLM_MODEL` in
  `.env` (and the default in `src/config.py` if it should change for everyone).
- **Tool-name collision with a model's own built-in tools.** With the web-search
  tool named `web_search`, `openai/gpt-oss-120b` called it with arguments shaped
  like `{"cursor": 2, "id": 0}` — the schema of gpt-oss's *own* built-in browsing
  tool, not ours — causing Groq to reject the call (`400 tool_use_failed`).
  Renaming it to `internet_search` and explicitly documenting in the tool's
  docstring that it takes a single plain `query: str` argument (not structured
  browser-style arguments) fixed it. Lesson: a tool name/shape that happens to
  match a model's own built-in tool can get silently confused with it.
- **`duckduckgo_search` → `ddgs`.** The package was renamed upstream; the old
  name now just emits a deprecation warning and re-exports the new one, but the
  search results themselves had also started coming back empty or
  locale-irrelevant (e.g. a login page for an unrelated local business) for
  some queries. Migrating the import to `ddgs` and passing `region="us-en"`
  fixed both the deprecation warning and the irrelevant-results problem.

(A few more issues came up in the same debugging session — the agent not
knowing the current date, an occasional runaway search loop, one bad tool call
being able to crash a whole batch evaluation run, and the model's citation
format not matching the prompt's — all fixed in `src/agent.py` and
`src/rag.py`; see their docstrings and inline comments for details.)

Running the actual RAGAS evaluation (`python -m src.evaluate --pipeline both`)
against Groq surfaced two more, specific to using a non-OpenAI judge model:

- **`answer_relevancy` requests `n=3`; Groq allows only `n=1`.** RAGAS's
  `AnswerRelevancy` metric generates 3 reverse-engineered questions per answer
  in a single call (`strictness=3`, passed as `n=3` to the LLM) to average
  over for robustness -- Groq rejects any `n>1` outright
  (`'n': number must be at most 1`). Fixed by constructing the metric with
  `strictness=1` in `src/evaluate.py`. This is a real tradeoff (one sampled
  question instead of three averaged), not a cosmetic workaround, and is
  purely a Groq-API constraint -- it wouldn't come up against OpenAI directly.
- **The judge ran out of output tokens mid-answer** (`LLMDidNotFinishException:
  generation was not completed`). Reasoning models like gpt-oss spend part of
  their output budget on internal reasoning before the actual answer, and
  ragas's default token budget assumption didn't leave enough room. Fixed by
  passing a higher explicit `max_tokens` (4096) for the judge LLM specifically
  (`src/rag.get_llm(..., max_tokens=...)`), without changing the model that
  answers questions.

Separately (not a bug, a capacity constraint worth knowing about): a full
full evaluation of one pipeline cost roughly 100-200k judge tokens, i.e.
**all of Groq's free-tier token quota (200k TPD, a rolling window per model)**
-- with the original five metrics it exhausted the quota before finishing a
single pipeline, so the two context metrics were dropped and only the three the
project reports are computed. Once the cap is hit, every further call 429s, and
ragas records `NaN` for those rather than crashing (`raise_exceptions=False`,
the default). Scoring is therefore done one question at a time with a per-question
cache (`data/eval_results/*_scores_cache.json`): only fully-scored questions are
cached, a run stops early after two consecutive questions with no scores, and a
re-run resumes where it stopped. (Also: `qwen/qwen3.8-27b` is unusable as a
judge on the free tier -- its 1,000 output-tokens-per-minute cap rejects any
request asking for `max_tokens=4096`.) The
evaluator's own retry layer is deliberately capped low (`RunConfig(max_retries=2)`
rather than ragas's default 10) specifically so hitting this doesn't also
balloon into dozens of doomed retry attempts per metric on top of the LLM
client's own 5 retries. If you hit this: switch `EVAL_LLM_MODEL` to a smaller
model (e.g. `openai/gpt-oss-20b`), evaluate `--pipeline rag` and
`--pipeline agent` as two separate runs (possibly on different days), or
upgrade the Groq account tier.


## Why base RAG refused 5 of 15 questions


**Follow-up fix, not yet re-measured.** Diagnosing those refusals showed the
right *document* was already ranked first for every question, but the specific
passage holding the answer often sat just outside the top 4 chunks (e.g. the FATF
CDD-measures passage ranks 8th for its question). Raising the default to
**top-k=8** made the base pipeline answer the CDD question (q02) and the PEP
close-family question (q05) that it previously refused; q07 and q10 still
refuse (q10's answer passage doesn't surface in the top 20 at all -- likely a
chunking/extraction issue with that PDF's text). The README table is the k=4
baseline; re-running `python -m src.evaluate --pipeline rag` will give the k=8
numbers once free-tier token quota allows (delete
`data/eval_results/rag_samples_cache.json` and `rag_scores_cache.json` first).
This was a spot check on the five failing questions, not a full re-evaluation.

## Why the agent column was re-run

An earlier attempt to score the agent was discarded: Groq's free-tier quota ran
out while answers were being generated, the agent's fallback message ("I couldn't
complete this request") was cached as if it were an answer, and scoring it gave
meaningless near-zero numbers. Failed agent runs now raise instead of being
cached.

A second issue was found later: the agent's knowledge-base contexts passed to
RAGAS were the tool outputs *as truncated for display* (first 1,000 characters
of ~5,600), so faithfulness was judged against only part of what the agent had
actually read. The agent now returns the full untruncated contexts separately,
which means agent answers cached before that fix must be regenerated.
