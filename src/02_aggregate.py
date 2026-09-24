"""
02_aggregate.py — агрегация таблиц МОБ до 5 отраслей.

Вход:  data/processed/
       ├── Z.npy              (243, 120)
       ├── X.npy              (120,)
       ├── I.npy              (243,)
       ├── C.npy              (243,)
       ├── codes_product.npy  (243,)
       └── codes_industry.npy (120,)

Выход: data/processed/
       ├── Z5.npy        (5, 5)   — промежуточное потребление по группам
       ├── X5.npy        (5,)     — выпуск по группам
       ├── I5.npy        (5,)     — валовое накопление по группам
       ├── C5.npy        (5,)     — конечное использование по группам
       ├── groups.json            — описание групп и размеров
       └── agg_meta.json          — суммы до и после агрегации

Группы (по префиксу ОКПД2):
    auto       — 29.*       (автотранспортные средства)
    metall     — 24.*, 25.* (металлургия + металлоизделия)
    fuel       — 05, 06, 19 (ТЭК: уголь, нефть/газ, кокс/нефтепродукты)
    transport  — 49–53      (все виды транспорта)
    other      — остальное
"""
from pathlib import Path
import json
import re
import numpy as np

IN  = Path('data/processed')
OUT = IN

GROUPS = ['auto', 'metall', 'fuel', 'transport', 'other']
G2I = {g: i for i, g in enumerate(GROUPS)}

GROUP_RULES = {
    'auto':      {29},
    'metall':    {24, 25},
    'fuel':      {5, 6, 19},
    'transport': {49, 50, 51, 52, 53},
}


def code_prefix(code):
    """Извлекает первые две цифры ОКПД2-кода."""
    m = re.match(r'\s*(\d{2})', str(code))
    return int(m.group(1)) if m else None


def code_to_group(code):
    """Относит код ОКПД2 к одной из 5 групп."""
    p = code_prefix(code)
    if p is None:
        return 'other'
    for group, prefixes in GROUP_RULES.items():
        if p in prefixes:
            return group
    return 'other'


def load(name):
    return np.load(IN / f'{name}.npy', allow_pickle=True)


# ============================================================
# 1. Загрузка
# ============================================================
Z              = load('Z')                # (243, 120)
X              = load('X')                # (120,)
I              = load('I')                # (243,)
C              = load('C')                # (243,)
codes_product  = load('codes_product')    # (243,)  строк
codes_industry = load('codes_industry')   # (120,)  строк

assert Z.shape == (243, 120)
assert X.shape == (120,)
assert I.shape == (243,)
assert C.shape == (243,)
assert codes_product.shape  == (243,)
assert codes_industry.shape == (120,)

# ============================================================
# 2. Карты принадлежности групп
# ============================================================
row_group = np.array([G2I[code_to_group(c)] for c in codes_product])   # (243,)
col_group = np.array([G2I[code_to_group(c)] for c in codes_industry])  # (120,)

print('Распределение продуктов по группам (строки Z, I, C):')
for g in GROUPS:
    n = int((row_group == G2I[g]).sum())
    print(f'  {g:10s}: {n:3d} продуктов')

print('\nРаспределение отраслей по группам (столбцы Z, X):')
for g in GROUPS:
    n = int((col_group == G2I[g]).sum())
    print(f'  {g:10s}: {n:3d} отраслей')

# ============================================================
# 3. Агрегация
# ============================================================
Z5 = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        mask = np.ix_(row_group == i, col_group == j)
        Z5[i, j] = Z[mask].sum()

X5 = np.array([X[col_group == i].sum() for i in range(5)])
I5 = np.array([I[row_group == i].sum() for i in range(5)])
C5 = np.array([C[row_group == i].sum() for i in range(5)])

# ============================================================
# 4. Проверки
# ============================================================
print('\nСуммы до и после агрегации:')
print(f'  Z   : {Z.sum():>15.0f}  →  Z5  : {Z5.sum():>15.0f}')
print(f'  X   : {X.sum():>15.0f}  →  X5  : {X5.sum():>15.0f}')
print(f'  I   : {I.sum():>15.0f}  →  I5  : {I5.sum():>15.0f}')
print(f'  C   : {C.sum():>15.0f}  →  C5  : {C5.sum():>15.0f}')

for name, a, b in [('Z', Z.sum(), Z5.sum()),
                   ('X', X.sum(), X5.sum()),
                   ('I', I.sum(), I5.sum()),
                   ('C', C.sum(), C5.sum())]:
    rel = abs(a - b) / a
    assert rel < 1e-9, f'{name}: агрегация исказила сумму на {rel*100:.4f}%'

# Проверка: доля авто в выпуске
auto_share = X5[G2I['auto']] / X5.sum()
print(f'\nДоля автомобилестроения в выпуске: {auto_share*100:.2f} %')

# Проверка: столбцовые суммы A (нормировка)
A5 = Z5 / X5[np.newaxis, :]
print('\nМатрица прямых затрат A5 (доли):')
print(np.round(A5, 4))
print('\nСуммы по столбцам A5 (промежуточное потребление / выпуск):')
for g in GROUPS:
    print(f'  {g:10s}: {A5[:, G2I[g]].sum():.4f}')

# ============================================================
# 5. Сохранение
# ============================================================
np.save(OUT / 'Z5.npy', Z5)
np.save(OUT / 'X5.npy', X5)
np.save(OUT / 'I5.npy', I5)
np.save(OUT / 'C5.npy', C5)

groups_info = {
    'names': GROUPS,
    'rules': {g: sorted(s) for g, s in GROUP_RULES.items()},
    'product_counts':  {g: int((row_group == G2I[g]).sum()) for g in GROUPS},
    'industry_counts': {g: int((col_group == G2I[g]).sum()) for g in GROUPS},
}

agg_meta = {
    'groups': groups_info,
    'sums_before': {
        'Z': float(Z.sum()), 'X': float(X.sum()),
        'I': float(I.sum()), 'C': float(C.sum()),
    },
    'sums_after': {
        'Z': float(Z5.sum()), 'X': float(X5.sum()),
        'I': float(I5.sum()), 'C': float(C5.sum()),
    },
    'auto_share_output': float(auto_share),
    'A5_column_sums': {g: float(A5[:, G2I[g]].sum()) for g in GROUPS},
}

(OUT / 'groups.json').write_text(
    json.dumps(groups_info, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'agg_meta.json').write_text(
    json.dumps(agg_meta, ensure_ascii=False, indent=2), encoding='utf-8')

print(f'\nГотово. Сохранено: Z5.npy, X5.npy, I5.npy, C5.npy, '
      f'groups.json, agg_meta.json')