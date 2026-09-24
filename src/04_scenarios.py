# 04_scenarios.py - три сценария с расширенным разбросом alpha, gamma.
#
# Модель: dK/dt = -M K + b(t)
#         X(t)  = G (C(t) + I_rec(t) - alpha K(t))
#         G = (I-A-alpha*beta)^{-1}
#         M = alpha^2 beta G + (alpha + delta) I
#         C(t)     = C0 - dU e1
#         I_rec(t) = gamma dU (1-exp(-lam t)) e_rec
#
# Сценарии:
#   growth  alpha=0.25 gamma=0.80  (быстрая адаптация + 80% возврата)
#   stag    alpha=0.10 gamma=0.30
#   crisis  alpha=0.02 gamma=0.00  (пассивность + полное изъятие)

import json
from pathlib import Path
import numpy as np

IN  = Path('data/processed')
OUT = IN

A     = np.load(IN / 'A.npy')
X5    = np.load(IN / 'X5.npy')
I5    = np.load(IN / 'I5.npy')
C5    = np.load(IN / 'C5.npy')
beta  = np.load(IN / 'beta.npy')
delta = float(np.load(IN / 'delta.npy')[0])
e_rec = np.load(IN / 'e_rec.npy')
B_eff = np.diag(beta)

DU_SHARE = 0.05
LAM      = 0.5
dU       = DU_SHARE * X5[0]
e1       = np.zeros(5); e1[0] = 1.0

print('=== Shok params ===')
print('dU = {:.0f} mln rub'.format(dU))
print('lam = {:.2f} 1/god'.format(LAM))
print('e_rec = ' + str(e_rec.tolist()))
print('')

SCENARIOS = {
    'growth': {'alpha': 0.25, 'gamma': 0.80},
    'stag':   {'alpha': 0.10, 'gamma': 0.30},
    'crisis': {'alpha': 0.02, 'gamma': 0.00},
}

results = {}
for name, p in SCENARIOS.items():
    alpha = p['alpha']
    gamma = p['gamma']

    G = np.linalg.inv(np.eye(5) - A - alpha * B_eff)
    M = alpha**2 * B_eff @ G + (alpha + delta) * np.eye(5)

    B_prime = alpha * delta / (alpha + delta) * B_eff
    C0 = (np.eye(5) - A - B_prime) @ X5
    K0 = alpha / (alpha + delta) * B_eff @ X5

    # Проверка калибровки
    X_check = G @ (C0 - alpha * K0)
    resid = float(np.linalg.norm(X_check - X5) / np.linalg.norm(X5))

    # Равновесие после шока
    C_inf     = C0 - dU * e1
    I_rec_inf = gamma * dU * e_rec
    b_inf     = alpha * B_eff @ G @ (C_inf + I_rec_inf) + I_rec_inf
    K_inf     = np.linalg.solve(M, b_inf)
    X_inf     = G @ (C_inf + I_rec_inf - alpha * K_inf)

    eig_M    = np.linalg.eigvals(M)
    min_re_M = float(np.min(eig_M.real))

    results[name] = {
        'alpha': alpha, 'gamma': gamma,
        'C0': C0.tolist(), 'K0': K0.tolist(),
        'K_inf': K_inf.tolist(), 'X_inf': X_inf.tolist(),
        'dX_rel': ((X_inf - X5) / X5).tolist(),
        'M_eig': [[float(e.real), float(e.imag)] for e in eig_M],
        'min_Re_mu_M': min_re_M,
        'calib_residual': resid,
    }

    print('=== {} ==='.format(name))
    print('alpha = {:.3f}, gamma = {:.3f}'.format(alpha, gamma))
    print('min Re mu(M) = {:+.4f}'.format(min_re_M))
    print('calib residual = {:.2e}'.format(resid))
    print('K0    = ' + str(np.round(K0, 0).tolist()))
    print('K_inf = ' + str(np.round(K_inf, 0).tolist()))
    print('X_inf = ' + str(np.round(X_inf, 0).tolist()))
    print('dX/X5 - 1, % = ' + str(np.round((X_inf / X5 - 1) * 100, 2).tolist()))
    print('')

    np.save(OUT / 'scen_{}_G.npy'.format(name), G)
    np.save(OUT / 'scen_{}_M.npy'.format(name), M)
    np.save(OUT / 'scen_{}_C0.npy'.format(name), C0)
    np.save(OUT / 'scen_{}_K0.npy'.format(name), K0)
    np.save(OUT / 'scen_{}_Kinf.npy'.format(name), K_inf)
    np.save(OUT / 'scen_{}_Xinf.npy'.format(name), X_inf)

(OUT / 'scenarios.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

print('Gotovo: scen_*_G.npy, scen_*_M.npy, ..., scenarios.json')