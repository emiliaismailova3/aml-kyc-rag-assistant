"""Tests for src/matching/: name normalization and the matching cascade.

A tiny character-trigram "embedding" replaces the real sentence-transformers
model so the tests are fast and deterministic.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.matching.evaluate import evaluate, score_predictions
from src.matching.matcher import Company, CompanyMatcher, MatchResult, load_companies
from src.matching.normalize import normalize_name

ROOT = Path(__file__).resolve().parent.parent


def fake_embed(texts):
    """Hash character trigrams into a 64-dim vector: similar spellings -> similar vectors."""
    vectors = np.zeros((len(texts), 64))
    for row, text in enumerate(texts):
        padded = f"  {text} "
        for i in range(len(padded) - 2):
            vectors[row, hash(padded[i : i + 3]) % 64] += 1
    return vectors.tolist()


COMPANIES = [
    Company(1, "Xəzər Logistika MMC", "1111111111"),
    Company(2, "Günəş Tikinti QSC", "2222222222"),
    Company(3, "İpək Telekom LLC", "3333333333"),
]


@pytest.fixture
def matcher():
    return CompanyMatcher(COMPANIES, embed_fn=fake_embed)


# --- normalization ------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"Xəzər Logistika" MMC', "xezer logistika"),
        ("XEZER LOGISTIKA MMC", "xezer logistika"),
        ("İpək Telekom LLC", "ipek telekom"),
        ("IPEK TELEKOM", "ipek telekom"),
        ("Şəki   Çörək, ASC.", "seki corek"),
        ("Gəncə-Ağdam Ltd", "gence agdam"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_normalize_can_keep_the_legal_form():
    assert normalize_name("Xəzər MMC", strip_legal_form=False) == "xezer mmc"


def test_dotted_capital_i_leaves_no_combining_dot():
    assert normalize_name("İİ") == "ii"


# --- matching cascade -------------------------------------------------------------

def test_voen_wins_even_if_the_name_is_wrong(matcher):
    result = matcher.match("Completely Different Name", voen="2222222222")
    assert (result.company_id, result.method, result.score) == (2, "voen", 1.0)


def test_unknown_voen_falls_back_to_the_name(matcher):
    result = matcher.match("Xəzər Logistika MMC", voen="9999999999")
    assert result.company_id == 1 and result.method == "exact_name"


def test_exact_name_ignores_case_diacritics_quotes_and_legal_form(matcher):
    result = matcher.match('"XEZER LOGISTIKA" LLC')
    assert result.company_id == 1 and result.method == "exact_name"


def test_fuzzy_catches_a_typo(matcher):
    result = matcher.match("Xezer Logsitika")
    assert result.company_id == 1
    assert result.method == "fuzzy" and not result.needs_human_review


def test_unrelated_name_is_not_matched(matcher):
    result = matcher.match("Sunrise Trading Ltd")
    assert result.company_id is None and result.method == "none"


def test_empty_name_is_not_matched(matcher):
    assert matcher.match("   ").company_id is None


def test_close_but_not_confident_goes_to_human_review():
    # The (fake) embedder rates the query 0.8 similar to every company: above the review
    # threshold (0.75) but below the automatic one (0.92), so a human must decide.
    def embed(texts):
        return [[0.8, 0.6]] if len(texts) == 1 else [[1.0, 0.0]] * len(texts)

    result = CompanyMatcher(COMPANIES, embed_fn=embed).match("Zzz Qqq Www")
    assert result.needs_human_review and result.company_id is not None
    assert result.method == "embedding" and result.score == 0.8


def test_embedding_step_is_used_when_fuzzy_fails():
    # Fuzzy string similarity is low, but the (fake) embedder says the names are identical.
    def embed(texts):
        return [[1.0, 0.0] if "alias" in t or "xezer" in t else [0.0, 1.0] for t in texts]

    matcher = CompanyMatcher(COMPANIES, embed_fn=embed)
    result = matcher.match("alias holding")
    assert result.company_id == 1 and result.method == "embedding"


# --- evaluation -----------------------------------------------------------------

def test_score_predictions_counts_precision_and_recall():
    pairs = [{"expected_id": 1}, {"expected_id": 2}, {"expected_id": None}, {"expected_id": 3}]
    results = [
        MatchResult(company_id=1),                          # correct
        MatchResult(company_id=9),                          # wrong
        MatchResult(company_id=4),                          # accepted but should be no match
        MatchResult(company_id=3, needs_human_review=True),  # sent to a human: not automatic
    ]
    stats = score_predictions(pairs, results)
    assert stats["accepted"] == 3 and stats["true_positives"] == 1
    assert stats["precision"] == round(1 / 3, 3)
    assert stats["recall"] == round(1 / 3, 3)
    assert stats["sent_to_human_review"] == 1


def test_labeled_pairs_file_is_consistent_with_the_company_table():
    companies = load_companies()
    ids = {c.id for c in companies}
    pairs = json.loads((ROOT / "data" / "reference" / "matching_pairs.json").read_text(encoding="utf-8"))
    assert len(companies) == 50 and len(pairs) >= 50
    assert all(p["expected_id"] is None or p["expected_id"] in ids for p in pairs)
    assert all(len(c.voen) == 10 and c.voen.isdigit() for c in companies)


def test_evaluate_on_the_real_pairs_with_the_fake_embedder_never_accepts_a_wrong_match():
    pairs = json.loads((ROOT / "data" / "reference" / "matching_pairs.json").read_text(encoding="utf-8"))
    report = evaluate(CompanyMatcher(load_companies(), embed_fn=fake_embed), pairs)
    assert report["false_positives"] == 0
    assert report["recall"] > 0.8
