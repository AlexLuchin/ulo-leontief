# 03_matrices.py - расширенная модель Леонтьева с гибким акселератором.
#
# Формулировка:
#   (I - A - alpha*beta) X = C + I_rec - alpha K
#   dK/dt = alpha (beta X - K) + I_rec - delta K
#
# Сводится к ODE по K:
#   dK/dt = -M K + b(t)
#   M = alpha^2 beta G + alpha + delta,   G = (I-A-alpha*beta)^{-1}
#   b(t) = alpha beta G (C(t) + I_rec(t)) + I_rec(t)
#
# Калибровка:
#   beta  = I5 / X5         (целевая капиталоёмкость из МОБ)
#   delta = 0.05            (норма амортизации)
#   alpha = 0.10            (скорость адаптации, << alpha*)
#
# Целевое инвестирование утильсбора:
#   I_rec(t) = gamma * dU * (1 - exp(-lam*t)) * e_rec
#   e_rec - вектор распределения (какие отрасли получают деньги)

import json
from pathlib import Path
import numpy as np

IN  = Path('data/processed')
OUT = IN

GROUPS = ['auto', 'metall', 'fuel', 'transport', 'other']

# ---- 1. Загрузка ----
A  = np.load(IN / 'A.npy')
X5 = np.load(IN / 'X5.npy')
I5 = np.load(IN / 'I5.npy')
C5 = np.load(IN / 'C5.npy')

# ---- 2. Калибровка beta, delta ----
beta  = I5 / X5            # вектор, длина 5
delta = 0.05               # скаляр

print('=== Kalibrovka ===')
print('beta  = I5/X5  = ' + str(np.round(beta, 4).tolist()))
print('delta = {:.3f}'.format(delta))

# ---- 3. Вектор распределения e_rec ----
# Металлургия 40%, транспорт 10%, прочие 50%.
# Авто и ТЭК не получают ничего.
e_rec = np.array([0.0, 0.4, 0.0, 0.1, 0.5])
print('e_rec = {} (auto/metall/fuel/transport/other)'.format(e_rec.tolist()))
np.save(OUT / 'e_rec.npy', e_rec)

# ---- 4. Поиск критического alpha* ----
def det_IA_ab(alpha):
    return np.linalg.det(np.eye(5) - A - alpha * np.diag(beta))

alphas = np.linspace(0.0, 5.0, 1001)
dets   = np.array([det_IA_ab(a) for a in alphas])

sign_changes = np.where(np.diff(np.sign(dets)) != 0)[0]
if len(sign_changes) > 0:
    i0 = sign_changes[0]
    alpha_crit = float(alphas[i0])
else:
    alpha_crit = None

print('')
print('=== Bifurkaciya ===')
if alpha_crit is not None:
    print('alpha* ~= {:.3f}'.format(alpha_crit))
else:
    print('det(I-A-alpha*beta) ne obrashaetsya v 0 na [0, 5]')

# ---- 5. Рабочая alpha и матрица M ----
alpha = 0.10
B_eff = np.diag(beta)
G = np.linalg.inv(np.eye(5) - A - alpha * B_eff)
M = alpha**2 * B_eff @ G + (alpha + delta) * np.eye(5)

eig_M = np.linalg.eigvals(M)
min_re_M = float(np.min(eig_M.real))

print('')
print('=== Rabochaya tochka: alpha = {:.3f} ==='.format(alpha))
print('Spectr M (dK/dt = -M K + b):')
for i, e in enumerate(sorted(eig_M, key=lambda x: x.real)):
    print('  mu_{} = {:+.4f} {:+.4f}i'.format(i + 1, e.real, e.imag))
print('min Re mu(M) = {:+.4f}'.format(min_re_M))
if min_re_M > 0:
    print('Status: USTOYCHIVO  (vse Re mu > 0 => -M ustoichiva)')
else:
    print('Status: NEUSTOYCHIVO (est Re mu < 0)')

# ---- 6. Проверка равновесия ----
X_star = np.linalg.solve(np.eye(5) - A, C5)
K_star = np.diag(beta) @ X_star
print('')
print('=== Ravnovesie bez shoka ===')
print('X*  = ' + str(np.round(X_star, 0).tolist()))
print('K*  = ' + str(np.round(K_star, 0).tolist()))

# ---- 7. Сохранение ----
np.save(OUT / 'beta.npy', beta)
np.save(OUT / 'delta.npy', np.array([delta]))
np.save(OUT / 'alpha.npy', np.array([alpha]))
np.save(OUT / 'M.npy', M)
np.save(OUT / 'G.npy', G)
np.save(OUT / 'alpha_scan.npy', alphas)
np.save(OUT / 'det_scan.npy', dets)

meta = {
    'model': 'Leontief + flexible accelerator + targeted recycling investment',
    'groups': GROUPS,
    'beta': beta.tolist(),
    'delta': delta,
    'alpha_working': alpha,
    'alpha_critical': alpha_crit,
    'e_rec': e_rec.tolist(),
    'M_eigenvalues': [[float(e.real), float(e.imag)] for e in eig_M],
    'min_Re_mu_M': min_re_M,
    'stable': bool(min_re_M > 0),
    'X_star': X_star.tolist(),
    'K_star': K_star.tolist(),
}
(OUT / 'matrices_meta.json').write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

print('')
print('Gotovo: beta.npy, delta.npy, alpha.npy, M.npy, G.npy,')
print('        e_rec.npy, alpha_scan.npy, det_scan.npy, matrices_meta.json')