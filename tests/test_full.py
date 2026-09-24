"""
Full-data tests — skipped if data/processed/*.npy are not available.

Проверяют:
  - суммы Z, X, I, C против опубликованных итогов Росстата,
  - A продуктивна, M устойчива,
  - порядок τ по сценариям (growth < stag < crisis),
  - знаки μ_K < 0 (сходимость),
  - X_inf > 0 (положительное равновесие).
"""
import json
from pathlib import Path
import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / 'data' / 'processed'


@pytest.mark.skipif(not (DATA / 'Z.npy').exists(),
                    reason='Z.npy not found — run 01_load.py first')
class TestLoadedData:
    """Проверки листа 01_load.py: суммы против итогов Росстата."""

    def test_Z_shape(self):
        Z = np.load(DATA / 'Z.npy')
        assert Z.shape == (243, 120), \
            'Z shape must be (243, 120), got {}'.format(Z.shape)

    def test_Z_sum(self):
        Z = np.load(DATA / 'Z.npy')
        expected = 124_282_711
        assert abs(Z.sum() - expected) < 1e3, \
            'Z sum = {:.0f}, expected {}'.format(Z.sum(), expected)

    def test_X_sum(self):
        X = np.load(DATA / 'X.npy')
        expected = 246_859_637
        assert abs(X.sum() - expected) < 1e3, \
            'X sum = {:.0f}, expected {}'.format(X.sum(), expected)

    def test_I_sum(self):
        I = np.load(DATA / 'I.npy')
        expected = 25_957_673
        assert abs(I.sum() - expected) < 1e3, \
            'I sum = {:.0f}, expected {}'.format(I.sum(), expected)

    def test_C_sum_tolerance(self):
        """C имеет известное расхождение 0.3% от опубликованного итога."""
        C = np.load(DATA / 'C.npy')
        expected = 162_459_521
        rel = abs(C.sum() - expected) / expected
        assert rel < 0.005, \
            'C sum deviation {:.3f}% > 0.5%'.format(rel * 100)


@pytest.mark.skipif(not (DATA / 'M.npy').exists(),
                    reason='M.npy not found — run 03_matrices.py first')
class TestMatrices:
    """Проверки листа 03_matrices.py: продуктивность, устойчивость."""

    def test_A_productive(self):
        A = np.load(DATA / 'A.npy')
        col_sums = A.sum(axis=0)
        assert np.all(col_sums < 1.0), \
            'A col sums must be < 1, got {}'.format(col_sums.tolist())

    def test_A_nonnegative(self):
        A = np.load(DATA / 'A.npy')
        assert np.all(A >= 0), 'A must be non-negative'

    def test_M_stable(self):
        M = np.load(DATA / 'M.npy')
        eig = np.linalg.eigvals(M)
        min_re = np.min(eig.real)
        assert min_re > 0, \
            'min Re mu(M) = {:.6f} must be > 0'.format(min_re)

    def test_beta_length(self):
        beta = np.load(DATA / 'beta.npy')
        assert beta.shape == (5,), \
            'beta shape must be (5,), got {}'.format(beta.shape)


@pytest.mark.skipif(not (DATA / 'scenarios.json').exists(),
                    reason='scenarios.json not found — run 04_scenarios.py first')
class TestScenarios:
    """Проверки листа 04_scenarios.py: три сценария, порядок τ."""

    def test_three_scenarios_present(self):
        s = json.loads((DATA / 'scenarios.json').read_text(encoding='utf-8'))
        assert set(s.keys()) == {'growth', 'stag', 'crisis'}, \
            'expected 3 scenarios, got {}'.format(set(s.keys()))

    def test_tau_ordering(self):
        """τ = 1/min Re μ(M) должно расти: growth < stag < crisis."""
        s = json.loads((DATA / 'scenarios.json').read_text(encoding='utf-8'))
        tau = {}
        for name, info in s.items():
            mu = info['min_Re_mu_M']
            tau[name] = 1.0 / mu
        assert tau['growth'] < tau['stag'] < tau['crisis'], \
            'tau ordering violated: {}'.format(tau)

    def test_X_inf_positive(self):
        s = json.loads((DATA / 'scenarios.json').read_text(encoding='utf-8'))
        for name, info in s.items():
            X_inf = np.array(info['X_inf'])
            assert np.all(X_inf > 0), \
                '{}: X_inf contains non-positive values'.format(name)

    def test_calibration_residual_small(self):
        s = json.loads((DATA / 'scenarios.json').read_text(encoding='utf-8'))
        for name, info in s.items():
            res = info['calib_residual']
            assert res < 1e-10, \
                '{}: calibration residual {:.2e} > 1e-10'.format(name, res)


@pytest.mark.skipif(not (DATA / 'integrate_summary.json').exists(),
                    reason='integrate_summary.json not found — run 05_integrate.py first')
class TestIntegration:
    """Проверки листа 05_integrate.py: знаки μ_K, положительность X_inf."""

    def test_mu_signs_negative(self):
        """Все μ_K < 0 — сходимость к равновесию во всех сценариях."""
        s = json.loads((DATA / 'integrate_summary.json')
                       .read_text(encoding='utf-8'))
        for name, info in s['scenarios'].items():
            assert info['mu_K'] < 0, \
                '{}: mu_K = {:.4f} must be negative'.format(name, info['mu_K'])

    def test_mu_K_matches_theory(self):
        """Численный μ_K должен совпадать с -min Re μ(M) в пределах 5%."""
        scen = json.loads((DATA / 'scenarios.json')
                          .read_text(encoding='utf-8'))
        integ = json.loads((DATA / 'integrate_summary.json')
                           .read_text(encoding='utf-8'))
        for name in ['growth', 'stag', 'crisis']:
            theory = -scen[name]['min_Re_mu_M']
            numeric = integ['scenarios'][name]['mu_K']
            rel = abs(numeric - theory) / abs(theory)
            assert rel < 0.05, \
                '{}: mu_K = {:.4f} vs theory {:.4f} ({:.2f}%)'.format(
                    name, numeric, theory, rel * 100)

    def test_X_inf_positive(self):
        s = json.loads((DATA / 'integrate_summary.json')
                       .read_text(encoding='utf-8'))
        for name, info in s['scenarios'].items():
            X_inf = np.array(info['X_inf'])
            assert np.all(X_inf > 0), \
                '{}: X_inf contains non-positive values'.format(name)

    def test_t_span_ordering(self):
        """Горизонт t_span должен быть минимальным для growth."""
        s = json.loads((DATA / 'integrate_summary.json')
                       .read_text(encoding='utf-8'))
        t_end = {}
        for name, info in s['scenarios'].items():
            t_end[name] = info['t_span'][1]
        assert t_end['growth'] <= t_end['stag'] <= t_end['crisis'], \
            't_span ordering violated: {}'.format(t_end)