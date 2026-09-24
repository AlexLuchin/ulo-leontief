# 06_visualize.py - визуализация расширенной модели.
#
# Вход:  data/processed/traj_*.npz, X5.npy, alpha_scan.npy, det_scan.npy
# Выход: out/scen_X.png, out/scen_K.png, out/scen_bars.png,
#        out/convergence.png, out/bifurcation.png,
#        out/adaptation_speed.png

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN  = Path('data/processed')
OUT = Path('out')
OUT.mkdir(exist_ok=True)

RUS    = ['Авто', 'Металл', 'ТЭК', 'Транспорт', 'Прочие']
COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#7f7f7f']
SCEN   = ['growth', 'stag', 'crisis']
TITLES = {'growth': 'Сценарий 1: Рост',
          'stag':   'Сценарий 2: Стагнация',
          'crisis': 'Сценарий 3: Падение'}
LSTYLE = {'growth': '-', 'stag': '--', 'crisis': ':'}

X5 = np.load(IN / 'X5.npy')

# ---- Загрузка траекторий ----
data = {}
for sc in SCEN:
    d = np.load(IN / 'traj_{}.npz'.format(sc), allow_pickle=True)
    data[sc] = {
        't':     d['t'],
        'K':     d['K'],
        'X':     d['X'],
        'K_inf': d['K_inf'],
        'X_inf': d['X_inf'],
        'mu_K':  float(d['mu_K']),
        'mu_X':  float(d['mu_X']),
        'alpha': float(d['alpha']),
        'gamma': float(d['gamma']),
    }

# ============================================================
# 1. X(t) / X5
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, sc in zip(axes, SCEN):
    d = data[sc]
    t = d['t']
    X = d['X']
    for i in range(5):
        ax.plot(t, X[i] / X5[i] * 100.0, color=COLORS[i],
                lw=1.8, ls=LSTYLE[sc], label=RUS[i])
    ax.axhline(100, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_title('{}\nalpha={:.2f}, gamma={:.2f}, mu_X={:+.3f}'.format(
        TITLES[sc], d['alpha'], d['gamma'], d['mu_X']), fontsize=10)
    ax.set_xlabel('Время t, годы')
    ax.set_ylabel('X(t) / X(0), %')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='best')
plt.tight_layout()
plt.savefig(OUT / 'scen_X.png', dpi=150)
plt.close()

# ============================================================
# 2. K(t) / K0
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, sc in zip(axes, SCEN):
    d = data[sc]
    t = d['t']
    K = d['K']
    K0 = K[:, 0]
    for i in range(5):
        if K0[i] > 1.0:
            ax.plot(t, K[i] / K0[i] * 100.0, color=COLORS[i],
                    lw=1.8, ls=LSTYLE[sc], label=RUS[i])
    ax.axhline(100, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_title('{} - kapital K(t)\nmu_K = {:+.3f}'.format(
        TITLES[sc], d['mu_K']), fontsize=10)
    ax.set_xlabel('Время t, годы')
    ax.set_ylabel('K(t) / K(0), %')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='best')
plt.tight_layout()
plt.savefig(OUT / 'scen_K.png', dpi=150)
plt.close()

# ============================================================
# 3. Bars: итоговое отклонение (X_inf - X5)/X5
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
width = 0.25
x = np.arange(5)
for idx, sc in enumerate(SCEN):
    d = data[sc]
    dX_rel = (d['X_inf'] - X5) / X5 * 100.0
    ax.bar(x + (idx - 1) * width, dX_rel, width,
           label='{} (tau ~ {:.1f} let)'.format(
               TITLES[sc], 1.0 / abs(d['mu_X'])))
ax.set_xticks(x)
ax.set_xticklabels(RUS)
ax.set_ylabel('(X_inf - X5) / X5, %')
ax.set_title('Итоговое отклонение выпуска vs скорость адаптации')
ax.axhline(0, color='k', lw=0.5)
ax.grid(alpha=0.3, axis='y')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / 'scen_bars.png', dpi=150)
plt.close()

# ============================================================
# 4. Сходимость K(t) к K_inf: нормированное отклонение в полулогарифме
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for sc in SCEN:
    d = data[sc]
    dev = d['K'] - d['K_inf'][:, np.newaxis]
    norm = np.linalg.norm(dev, axis=0)
    dev0 = norm[0] if norm[0] > 0 else 1.0
    norm_rel = norm / dev0

    mask = norm_rel > 1e-9
    t_m   = d['t'][mask]
    n_m   = norm_rel[mask]

    slope = float(np.polyfit(t_m, np.log(n_m), 1)[0]) if mask.sum() > 5 else 0.0
    tau = 1.0 / abs(slope) if slope != 0.0 else float('inf')

    ax.semilogy(t_m, n_m, lw=1.6,
                label='{}: slope={:+.3f} (tau={:.1f})'.format(
                    TITLES[sc], slope, tau))

ax.axhline(1.0, color='k', lw=0.5, ls='--', alpha=0.4)
ax.set_xlabel('Время t, годы')
ax.set_ylabel('||K(t) - K_inf|| / ||K(0) - K_inf||')
ax.set_title('Сходимость капитала к новому равновесию\n'
             '(нормированное отклонение, полулогарифм)')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / 'convergence.png', dpi=150)
plt.close()

# ============================================================
# 5. Бифуркационная диаграмма
# ============================================================
alpha_scan = np.load(IN / 'alpha_scan.npy')
det_scan   = np.load(IN / 'det_scan.npy')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(alpha_scan, det_scan, lw=1.6, color='k')
ax.axhline(0, color='k', lw=0.5)
for sc in SCEN:
    a_sc = data[sc]['alpha']
    ax.axvline(a_sc, color=COLORS[SCEN.index(sc)], ls='--', lw=1.4,
               label='{}: alpha={:.2f}'.format(TITLES[sc], a_sc))
ax.set_xlabel('alpha (skorost adaptacii)')
ax.set_ylabel('det(I - A - alpha*beta)')
ax.set_title('Бифуркационная диаграмма: det = 0 при alpha* ~ 1.25')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
ax.set_xlim(0, 2.0)
plt.tight_layout()
plt.savefig(OUT / 'bifurcation.png', dpi=150)
plt.close()

# ============================================================
# 6. Кривая скорости адаптации: нормированное отклонение в полулогарифме
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for sc in SCEN:
    d = data[sc]
    t = d['t']
    X_auto     = d['X'][0]
    X_inf_auto = d['X_inf'][0]

    dev = np.abs(X_auto - X_inf_auto)
    dev0 = dev[0] if dev[0] > 0 else 1.0
    dev_norm = dev / dev0           # нормируем: все стартуют с 1.0

    mask = dev_norm > 1e-9
    t_m = t[mask]
    dev_m = dev_norm[mask]

    # Наклон регрессии log(dev_norm) на t = скорость адаптации
    if mask.sum() > 5:
        slope = float(np.polyfit(t_m, np.log(dev_m), 1)[0])
    else:
        slope = 0.0
    tau = 1.0 / abs(slope) if slope != 0.0 else float('inf')

    ax.semilogy(t_m, dev_m, color=COLORS[SCEN.index(sc)],
                lw=1.8, ls=LSTYLE[sc],
                label='{}: slope={:+.3f} (tau={:.1f} let)'.format(
                    TITLES[sc], slope, tau))

ax.axhline(1.0, color='k', lw=0.5, ls='--', alpha=0.4)
ax.set_xlabel('Время t, годы')
ax.set_ylabel('|X_auto(t) - X_inf| / |X_auto(0) - X_inf|')
ax.set_title('Кривая скорости адаптации автомобилестроения\n'
             '(нормированное отклонение от нового равновесия, полулогарифм)')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig(OUT / 'adaptation_speed.png', dpi=150)
plt.close()

# ============================================================
# 7. Нормированное отклонение авто на линейной оси
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for sc in SCEN:
    d = data[sc]
    t = d['t']
    X_auto     = d['X'][0]
    X_inf_auto = d['X_inf'][0]

    dev = X_auto - X_inf_auto
    dev0 = dev[0] if abs(dev[0]) > 0 else 1.0
    dev_norm = dev / dev0

    tau = 1.0 / abs(d['mu_X'])

    ax.plot(t / tau, dev_norm, color=COLORS[SCEN.index(sc)],
            lw=2.0, ls=LSTYLE[sc],
            label='{}: tau={:.1f} let'.format(TITLES[sc], tau))

ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
ax.axhline(1, color='k', lw=0.5, ls='--', alpha=0.5)
ax.axvline(1, color='grey', lw=0.5, ls=':', alpha=0.5)
ax.set_xlim(0, 6)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('Нормированное время $t/\\tau$')
ax.set_ylabel('$(X_{auto}(t) - X_{inf}) / (X_{auto}(0) - X_{inf})$')
ax.set_title('Нормированное отклонение автомобилестроения (линейная ось)\n'
             'Все три сценария коллапсируют на одну кривую $e^{-t/\\tau}$')
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(OUT / 'adaptation_collapse.png', dpi=150)
plt.close()

print('  out/adaptation_collapse.png - нормированное отклонение (линейная ось)')

print('Gotovo:')
print('  out/scen_X.png           - X(t)/X(0), tri scenariya')
print('  out/scen_K.png           - K(t)/K(0)')
print('  out/scen_bars.png        - itogovoe otklonenie')
print('  out/convergence.png      - ||K(t)-K_inf||')
print('  out/bifurcation.png      - det(I-A-alpha*beta)')
print('  out/adaptation_speed.png - |X_auto(t) - X_inf_auto| (semilog)')