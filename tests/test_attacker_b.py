import numpy as np

from src.attacks import _semantic_guess


def test_unique_continuous_is_not_identifier():
    rng = np.random.RandomState(0)
    col = rng.normal(size=400)
    labels = [g["label"] for g in _semantic_guess(col, "generic table")]
    assert "identifier" not in labels


def test_age_like_column_is_guessed():
    col = np.clip(np.random.RandomState(1).normal(42, 12, size=500), 18, 90).round()
    labels = [g["label"] for g in _semantic_guess(col, "bank customer statistics")]
    assert "age" in labels


def test_skewed_positive_looks_like_money_in_bank_context():
    col = np.exp(np.random.RandomState(2).normal(10, 0.5, size=500))
    labels = [g["label"] for g in _semantic_guess(col, "This dataset comes from a bank")]
    assert any(x in labels for x in {"money_or_income", "bank_amount"})
