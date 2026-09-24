# 05_integrate.py - BDF-интегрирование расширенной модели Леонтьева.
#
# Модель:
#   (I - A - alpha*beta) X = C(t) + I_rec(t) - alpha K
#   dK/dt = alpha (beta X - K) + I_rec(t) - delta K
#
# Сводится к ODE по K:
#   dK/dt = -M K + b(t)
#   M     = alpha^2 beta G + (alpha + delta) I
#   G     = (I - A - alpha*beta)^{-1}
#   b(t)  = alpha beta G (C(t) + I_rec(t)) + I_rec(t)
#
# Восстановление выпуска:
#   X(t) = G (C(t) + I_rec(t) - alpha K(t))
#
# Шок:
#   C(t)      = C0 - dU * e1
#   I_rec(t)  = gamma * dU * (1 - exp(-lam*t)) * e_rec
#
# Три сценария различаются (alpha, gamma) и горизонтом T_SPAN_MAP:
#   growth - alpha=0.25, gamma=0.80, t до 30 лет   (8*tau)
#   stag   - alpha=0.10, gamma=0.30, t до 60 лет   (8*tau)
#   crisis - alpha=0.02, gamma=0.00, t до 150 лет  (10*tau)
#
# Оценка mu_K и mu_X: фит log|dev| выполняется ТОЛЬКО на асимптотическом
# участке (t > 1.5*tau), где доминирует одно собственное значение.
# Это устраняет смещение фита на коротких горизонтах.
#
# Метод: scipy.integrate.solve_ivp(method='BDF'), L-устойчивый.

import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

IN  = Path('data/processed')
OUT = IN

SCENARIO_NAMES = ['growth', 'stag', 'crisis']

# Горизонт зависит от скорости сценария
T_SPAN_MAP = {
    'growth': (0.0,  30.0),   # ~ 8*tau (tau = 3.3)
    'stag':   (0.0,  60.0),   # ~ 8*tau (tau = 6.7)
    'crisis': (0.0, 150.0),   # ~ 10*tau (tau = 14.3)
}

DU_SHARE = 0.05
LAM      = 0.5

# ---- 1. Загрузка общих данных ----
A     = np.load(IN / 'A.npy')
X5    = np.load(IN / 'X5.npy')
beta  = np.load(IN / 'beta.npy')
delta = float(np.load(IN / 'delta.npy')[0])
e_rec = np.load(IN / 'e_rec.npy')
B_eff = np.diag(beta)

dU = DU_SHARE * X5[0]
e1 = np.zeros(5); e1[0] = 1.0

# ---- 2. Загрузка сценариев ----
scen_data = {}
for name in SCENARIO_NAMES:
    scen_data[name] = {
        'G':    np.load(IN / 'scen_{}_G.npy'.format(name)),
        'M':    np.load(IN / 'scen_{}_M.npy'.format(name)),
        'C0':   np.load(IN / 'scen_{}_C0.npy'.format(name)),
        'K0':   np.load(IN / 'scen_{}_K0.npy'.format(name)),
        'Kinf': np.load(IN / 'scen_{}_Kinf.npy'.format(name)),
        'Xinf': np.load(IN / 'scen_{}_Xinf.npy'.format(name)),
    }

scen_json = json.loads((IN / 'scenarios.json').read_text(encoding='utf-8'))


# ---- 3. Функции шока и правой части ----
def C_of_t(t, C0):
    # C(t) = C0 - dU * e1   (шок мгновенный, потом постоянно)
    return C0 - dU * e1


def Irec_of_t(t, gamma):
    # I_rec(t) = gamma dU (1 - exp(-lam t)) e_rec
    if gamma <= 0.0 or t <= 0.0:
        return np.zeros(5)
    return gamma * dU * (1.0 - np.exp(-LAM * t)) * e_rec


def make_rhs(G, M, C0, gamma, alpha):
    def rhs(t, K):
        C    = C_of_t(t, C0)
        Irec = Irec_of_t(t, gamma)
        b    = alpha * (B_eff @ (G @ (C + Irec))) + Irec
        return -M @ K + b
    return rhs


# ---- 4. Интегрирование по сценариям ----
results = {}

for name in SCENARIO_NAMES:
    d = scen_data[name]
    G, M, C0, K0, Kinf, Xinf = (d['G'], d['M'], d['C0'],
                                d['K0'], d['Kinf'], d['Xinf'])
    alpha = scen_json[name]['alpha']
    gamma = scen_json[name]['gamma']
    t_span = T_SPAN_MAP[name]
    min_re_mu = scen_json[name]['min_Re_mu_M']
    tau_est   = 1.0 / min_re_mu
    t_fit_min = 1.5 * tau_est   # асимптотический участок: t > 1.5*tau

    rhs = make_rhs(G, M, C0, gamma, alpha)

    sol = solve_ivp(
        fun=rhs,
        t_span=t_span,
        y0=K0,
        method='BDF',
        rtol=1e-9,
        atol=1e-9,
        dense_output=False,
        max_step=0.5,
    )

    if sol.status != 0:
        print('WARN: scenario {} status={}'.format(name, sol.status))

    # Восстановление X(t) = G (C(t) + I_rec(t) - alpha K(t))
    n_t = sol.t.size
    X_t = np.zeros((5, n_t))
    for j in range(n_t):
        t_j   = sol.t[j]
        K_j   = sol.y[:, j]
        C_j   = C_of_t(t_j, C0)
        I_j   = Irec_of_t(t_j, gamma)
        X_t[:, j] = G @ (C_j + I_j - alpha * K_j)

    # ---- Скорость сходимости по K ----
    # Фит только на асимптотическом участке t > 1.5*tau
    dev_K    = sol.y - Kinf[:, np.newaxis]
    dev_K_n  = np.linalg.norm(dev_K, axis=0)
    dev_K_0  = dev_K_n[0] if dev_K_n[0] > 0 else 1.0
    dev_K_r  = dev_K_n / dev_K_0

    mask_K = (dev_K_r > 1e-9) & (sol.t > t_fit_min)
    if mask_K.sum() > 10:
        mu_K = float(np.polyfit(sol.t[mask_K],
                                np.log(dev_K_r[mask_K]), 1)[0])
    else:
        mu_K = 0.0

    # ---- Скорость сходимости по X ----
    dev_X    = X_t - Xinf[:, np.newaxis]
    dev_X_n  = np.linalg.norm(dev_X, axis=0)
    dev_X_0  = dev_X_n[0] if dev_X_n[0] > 0 else 1.0
    dev_X_r  = dev_X_n / dev_X_0

    mask_X = (dev_X_r > 1e-9) & (sol.t > t_fit_min)
    if mask_X.sum() > 10:
        mu_X = float(np.polyfit(sol.t[mask_X],
                                np.log(dev_X_r[mask_X]), 1)[0])
    else:
        mu_X = 0.0

    results[name] = {
        't': sol.t.tolist(),
        'K': sol.y.tolist(),
        'X': X_t.tolist(),
        'K_inf': Kinf.tolist(),
        'X_inf': Xinf.tolist(),
        'mu_K': mu_K,
        'mu_X': mu_X,
        'alpha': float(alpha),
        'gamma': float(gamma),
        't_span': list(t_span),
        't_fit_min': float(t_fit_min),
        'status': int(sol.status),
        'n_steps': int(sol.t.size),
    }

    print('=== {} ==='.format(name))
    print('alpha={:.3f}, gamma={:.3f}, t_span={}'.format(
        alpha, gamma, list(t_span)))
    print('min Re mu(M) = {:+.4f}  ->  tau = {:.2f} let'.format(
        min_re_mu, tau_est))
    print('t_fit_min    = {:.2f} let (1.5*tau)'.format(t_fit_min))
    print('mu_K (fit)   = {:+.4f}   (theory: {:+.4f}, rel: {:.2f}%)'.format(
        mu_K, -min_re_mu,
        100.0 * abs(mu_K + min_re_mu) / min_re_mu))
    print('mu_X (fit)   = {:+.4f}'.format(mu_X))
    print('steps        = {}'.format(sol.t.size))
    print('X(t_end)[auto]/X5[auto] = {:.4f}'.format(X_t[0, -1] / X5[0]))
    print('')

    np.savez(
        OUT / 'traj_{}.npz'.format(name),
        t=sol.t, K=sol.y, X=X_t,
        K_inf=Kinf, X_inf=Xinf,
        mu_K=mu_K, mu_X=mu_X,
        dU=dU, gamma=gamma, alpha=alpha,
        t_span=np.array(t_span),
        t_fit_min=t_fit_min,
    )

# ---- 5. Сводка ----
(OUT / 'integrate_summary.json').write_text(
    json.dumps({
        'params': {'dU': float(dU), 'lam': LAM},
        't_span_map': {k: list(v) for k, v in T_SPAN_MAP.items()},
        'scenarios': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

print('Gotovo: traj_growth.npz, traj_stag.npz, traj_crisis.npz,')
print('        integrate_summary.json')