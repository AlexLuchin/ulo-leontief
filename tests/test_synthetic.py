"""
Invariant tests on a small synthetic model — run everywhere, no data needed.

Проверяют математические свойства модели:
  1. A продуктивна (суммы по столбцам < 1).
  2. G = (I-A-αβ)^{-1} обратима и не вырождена.
  3. M = α²βG + (α+δ)I устойчива: min Re μ(M) > 0.
  4. Критическое α* > 0 (точка бифуркации существует).
  5. Калибровка X5 = G(C0 - α K0) воспроизводится с точностью 1e-10.
"""
import numpy as np
import pytest


def test_A_is_productive(synthetic_model):
    """A должна быть продуктивной: суммы по столбцам < 1."""
    A = synthetic_model['A']
    col_sums = A.sum(axis=0)
    assert np.all(col_sums < 1.0), \
        'A must be productive (col sums < 1), got {}'.format(col_sums.tolist())
    assert np.all(A >= 0), 'A must be non-negative'


def test_G_invertible(synthetic_model):
    """G = (I-A-αβ)^{-1} должна быть обратимой."""
    G = synthetic_model['G']
    assert np.isfinite(G).all(), 'G contains non-finite values'
    assert np.linalg.cond(G) < 1e6, \
        'G is ill-conditioned (cond = {:.2e})'.format(np.linalg.cond(G))


def test_M_is_stable(synthetic_model):
    """
    M должна иметь все собственные значения с Re μ > 0.
    Это означает, что -M устойчива (все Re < 0) и система сходится.
    """
    M = synthetic_model['M']
    eig = np.linalg.eigvals(M)
    min_re = np.min(eig.real)
    assert min_re > 0, \
        'min Re mu(M) must be > 0 for stability, got {:.6f}'.format(min_re)


def test_bifurcation_alpha_positive(synthetic_model):
    """Критическое alpha* должно быть > 0 (бифуркация существует)."""
    A    = synthetic_model['A']
    beta = synthetic_model['beta']
    n    = synthetic_model['n']

    def det_IA_ab(a):
        return np.linalg.det(np.eye(n) - A - a * np.diag(beta))

    alphas = np.linspace(0.0, 5.0, 501)
    dets = np.array([det_IA_ab(a) for a in alphas])

    sign_changes = np.where(np.diff(np.sign(dets)) != 0)[0]
    assert len(sign_changes) > 0, 'bifurcation point must exist'
    alpha_star = alphas[sign_changes[0]]
    assert alpha_star > 0.0, \
        'critical alpha* must be > 0, got {:.4f}'.format(alpha_star)


def test_calibration_residual():
    """
    Проверка калибровки: X5 = G(C0 - α K0) должна воспроизводиться
    с точностью 1e-10 на случайной синтетической модели.
    """
    rng = np.random.default_rng(42)
    n = 3

    A = rng.uniform(0, 0.15, (n, n))
    for j in range(n):
        A[:, j] = A[:, j] / A[:, j].sum() * 0.5

    beta  = rng.uniform(0.1, 0.5, n)
    delta = 0.05
    alpha = 0.10
    B_eff = np.diag(beta)

    G = np.linalg.inv(np.eye(n) - A - alpha * B_eff)
    X5 = rng.uniform(1e6, 1e7, n)

    B_prime = alpha * delta / (alpha + delta) * B_eff
    C0 = (np.eye(n) - A - B_prime) @ X5
    K0 = alpha / (alpha + delta) * B_eff @ X5

    X_check = G @ (C0 - alpha * K0)
    rel = np.linalg.norm(X_check - X5) / np.linalg.norm(X5)
    assert rel < 1e-10, \
        'calibration residual too large: {:.2e}'.format(rel)