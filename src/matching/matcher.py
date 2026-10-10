"""Company matching: exact VOEN -> exact name -> fuzzy -> embedding similarity.

Each step is cheaper and more certain than the next, so we stop at the first
one that is confident enough. If nothing is confident but something is close,
the result is flagged for a human instead of guessing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from rapidfuzz import fuzz

from src.config import PROJECT_ROOT
from src.matching.normalize import normalize_name

COMPANIES_PATH = PROJECT_ROOT / "data" / "reference" / "companies.json"

# Thresholds were chosen from the precision/recall check in src/matching/evaluate.py
# (see docs/DECISIONS.md). At or above ACCEPT the match is used automatically;
# between REVIEW and ACCEPT a human decides; below REVIEW there is no match.
FUZZY_ACCEPT = 0.90
EMBEDDING_ACCEPT = 0.92
REVIEW_THRESHOLD = 0.75

# texts -> one vector per text. Injected so tests do not load a real model.
EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class Company:
    id: int
    name: str
    voen: str


class MatchResult(BaseModel):
    company_id: int | None = None
    company_name: str | None = None
    score: float = 0.0
    method: str = "none"  # voen | exact_name | fuzzy | embedding | none
    needs_human_review: bool = False


def load_companies(path: Path = COMPANIES_PATH) -> list[Company]:
    return [Company(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def _default_embed(texts: list[str]) -> list[list[float]]:
    from src.vectorstore import get_embeddings

    return get_embeddings().embed_documents(texts)


class CompanyMatcher:
    def __init__(self, companies: list[Company], embed_fn: EmbedFn | None = None):
        self.companies = companies
        self._embed_fn = embed_fn or _default_embed
        self._names = [normalize_name(c.name) for c in companies]
        self._by_voen = {c.voen: c for c in companies}
        self._by_name = dict(zip(self._names, companies, strict=True))
        self._vectors: np.ndarray | None = None  # computed lazily: embedding is the slow step

    def _name_vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = _unit(np.array(self._embed_fn(self._names), dtype=float))
        return self._vectors

    def match(self, name: str, voen: str | None = None) -> MatchResult:
        # 1. A matching VOEN is a legal identifier: certain.
        if voen and voen.strip() in self._by_voen:
            return _result(self._by_voen[voen.strip()], 1.0, "voen")

        query = normalize_name(name)
        if not query:
            return MatchResult()

        # 2. Same name after normalization (case, diacritics, quotes, legal form).
        if query in self._by_name:
            return _result(self._by_name[query], 1.0, "exact_name")

        # 3. Fuzzy string similarity: typos, small differences.
        scores = [fuzz.token_sort_ratio(query, n) / 100 for n in self._names]
        best = int(np.argmax(scores))
        fuzzy_score = scores[best]
        if fuzzy_score >= FUZZY_ACCEPT:
            return _result(self.companies[best], fuzzy_score, "fuzzy")

        # 4. Embedding similarity: catches what the character-based score misses.
        query_vector = _unit(np.array(self._embed_fn([query]), dtype=float))[0]
        similarities = self._name_vectors() @ query_vector
        best_emb = int(np.argmax(similarities))
        embedding_score = float(similarities[best_emb])
        if embedding_score >= EMBEDDING_ACCEPT:
            return _result(self.companies[best_emb], embedding_score, "embedding")

        # 5. Nothing confident. Close enough to be worth a human look?
        score, method, index = max([(fuzzy_score, "fuzzy", best), (embedding_score, "embedding", best_emb)])
        if score >= REVIEW_THRESHOLD:
            result = _result(self.companies[index], score, method)
            result.needs_human_review = True
            return result
        return MatchResult(score=round(score, 3))


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def _result(company: Company, score: float, method: str) -> MatchResult:
    return MatchResult(company_id=company.id, company_name=company.name, score=round(score, 3), method=method)
