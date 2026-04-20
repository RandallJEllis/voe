"""Tests for QR core operations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import solve_triangular

from regsens.givens import givens_params, apply_givens_rows, apply_givens_cols
from regsens.qr_core import (
    QRState, qr_state_from_cols, qr_add_column, qr_remove_column,
    qr_new_outcome, qr_batch_outcomes, compute_se,
)

RNG = np.random.default_rng(0)
PAIRS = [(50, 3), (100, 8), (30, 5)]


# ---------------------------------------------------------------------------
# Givens
# ---------------------------------------------------------------------------

def test_givens_zero_b():
    c, s, r = givens_params(3.0, 0.0)
    assert c == pytest.approx(1.0)
    assert s == pytest.approx(0.0)
    assert r == pytest.approx(3.0)


def test_givens_zero_a():
    c, s, r = givens_params(0.0, 4.0)
    assert c == pytest.approx(0.0)
    assert s == pytest.approx(1.0)
    assert r == pytest.approx(4.0)


@pytest.mark.parametrize("a, b", [(3.0, 4.0), (-1.5, 2.0), (0.1, -0.9), (100.0, 0.001)])
def test_givens_zeros_second_component(a, b):
    c, s, r = givens_params(a, b)
    result = np.array([[c, s], [-s, c]]) @ np.array([a, b])
    assert result[1] == pytest.approx(0.0, abs=1e-12)
    assert result[0] == pytest.approx(r, abs=1e-12)


@pytest.mark.parametrize("m, p", PAIRS)
def test_qr_state_orthogonality(m, p):
    X = RNG.standard_normal((m, p))
    state = qr_state_from_cols(X, list(range(p)))
    assert_allclose(state.Q.T @ state.Q, np.eye(p), atol=1e-12)


@pytest.mark.parametrize("m, p", PAIRS)
def test_qr_state_factorization(m, p):
    X = RNG.standard_normal((m, p))
    state = qr_state_from_cols(X, list(range(p)))
    assert_allclose(state.Q @ state.R, X, atol=1e-10)


@pytest.mark.parametrize("m, p", PAIRS)
def test_qr_state_positive_diagonal(m, p):
    X = RNG.standard_normal((m, p))
    state = qr_state_from_cols(X, list(range(p)))
    assert np.all(np.diag(state.R) >= 0)


@pytest.mark.parametrize("m, p", PAIRS)
def test_add_column_orthogonality(m, p):
    X = RNG.standard_normal((m, p + 1))
    state = qr_state_from_cols(X, list(range(p)))
    state = qr_add_column(state, X[:, p], p, inplace=False)
    assert_allclose(state.Q.T @ state.Q, np.eye(p + 1), atol=1e-12)


@pytest.mark.parametrize("m, p", PAIRS)
def test_add_column_factorization(m, p):
    X = RNG.standard_normal((m, p + 1))
    state = qr_state_from_cols(X, list(range(p)))
    state = qr_add_column(state, X[:, p], p, inplace=False)
    assert_allclose(state.Q @ state.R, X, atol=1e-10)


@pytest.mark.parametrize("m, p", PAIRS)
def test_add_column_matches_fresh_qr(m, p):
    X = RNG.standard_normal((m, p + 1))
    state = qr_state_from_cols(X, list(range(p)))
    state_add = qr_add_column(state.copy(), X[:, p], p, inplace=True)
    state_fresh = qr_state_from_cols(X, list(range(p + 1)))
    # QR is unique up to sign; compare X = Q R directly
    assert_allclose(state_add.Q @ state_add.R, state_fresh.Q @ state_fresh.R, atol=1e-10)


@pytest.mark.parametrize("pos", [0, -1, "mid"])
def test_remove_column_orthogonality(pos):
    m, p = 60, 5
    X = RNG.standard_normal((m, p))
    state = qr_state_from_cols(X, list(range(p)))
    remove_pos = 0 if pos == 0 else (p - 1 if pos == -1 else p // 2)
    state = qr_remove_column(state, remove_pos, inplace=False)
    assert_allclose(state.Q.T @ state.Q, np.eye(p - 1), atol=1e-12)


@pytest.mark.parametrize("pos", [0, -1, "mid"])
def test_remove_column_factorization(pos):
    m, p = 60, 5
    X = RNG.standard_normal((m, p))
    state = qr_state_from_cols(X, list(range(p)))
    remove_pos = 0 if pos == 0 else (p - 1 if pos == -1 else p // 2)
    remaining = [j for j in range(p) if j != remove_pos]
    state_rm = qr_remove_column(state, remove_pos, inplace=False)
    assert_allclose(state_rm.Q @ state_rm.R, X[:, remaining], atol=1e-10)


def test_add_then_remove_roundtrip():
    m, p = 50, 4
    X = RNG.standard_normal((m, p + 1))
    state0 = qr_state_from_cols(X, list(range(p)))
    state1 = qr_add_column(state0.copy(), X[:, p], p, inplace=True)
    state2 = qr_remove_column(state1, p, inplace=True)  # remove the last col
    # factorizations must agree
    assert_allclose(state2.Q @ state2.R, state0.Q @ state0.R, atol=1e-10)


@pytest.mark.parametrize("m, p", PAIRS)
def test_qr_new_outcome_matches_lstsq(m, p):
    X = RNG.standard_normal((m, p))
    y = RNG.standard_normal(m)
    state = qr_state_from_cols(X, list(range(p)))
    fit = qr_new_outcome(state, y)
    ref_beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    assert_allclose(fit["beta"], ref_beta, atol=1e-8)


@pytest.mark.parametrize("m, p", PAIRS)
def test_qr_new_outcome_rss_correct(m, p):
    X = RNG.standard_normal((m, p))
    y = RNG.standard_normal(m)
    state = qr_state_from_cols(X, list(range(p)))
    fit = qr_new_outcome(state, y)
    expected_rss = float(np.sum((y - X @ fit["beta"]) ** 2))
    assert fit["rss"] == pytest.approx(expected_rss, rel=1e-8)


def test_qr_batch_outcomes_matches_single():
    m, p, q = 50, 4, 5
    X = RNG.standard_normal((m, p))
    Y = RNG.standard_normal((m, q))
    state = qr_state_from_cols(X, list(range(p)))
    batch = qr_batch_outcomes(state, Y)
    for qi in range(q):
        single = qr_new_outcome(state, Y[:, qi])
        assert_allclose(batch["beta"][:, qi], single["beta"], atol=1e-10)
        assert batch["rss"][qi] == pytest.approx(single["rss"], rel=1e-8)


def test_compute_se_matches_manual():
    m, p = 50, 4
    X = RNG.standard_normal((m, p))
    y = RNG.standard_normal(m)
    state = qr_state_from_cols(X, list(range(p)))
    fit = qr_new_outcome(state, y)
    sigma = float(np.sqrt(fit["rss"] / (m - p)))
    se = compute_se(state.R, sigma)
    # Reference: sigma * sqrt(diag(inv(X.T @ X)))
    XtX_inv = np.linalg.inv(X.T @ X)
    se_ref = sigma * np.sqrt(np.diag(XtX_inv))
    assert_allclose(se, se_ref, atol=1e-8)
