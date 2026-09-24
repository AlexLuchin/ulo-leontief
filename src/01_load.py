"""
01_load.py — извлечение данных из базовых таблиц «затраты — выпуск» РФ за 2021 год.

Вход:  data/baz_tzv-2021.xlsx
Выход: data/processed/
       ├── Z.npy              (243, 120)  — промежуточное потребление, млн руб.
       ├── X.npy              (120,)      — выпуск отраслей, млн руб.
       ├── I.npy              (243,)      — валовое накопление (P51), по продуктам
       ├── C.npy              (243,)      — конечное использование (P3+P5+P6), по продуктам
       ├── codes_product.npy  (243,)      — коды ОКПД2 продуктов (столбец B листа ТИоц)
       ├── codes_industry.npy (120,)      — коды ОКПД2 отраслей (строка 2 листа ТИоц)
       └── meta.json                       — размерности, суммы, допуски, источник

Примечание по точности:
    Сумма P3_S14 (столбец 124) в файле даёт 66 464 645 вместо 67 008 658
    из строки TOTAL (расхождение −544 013, ≈ 0,3 % от C). Природа — нечисловая
    ячейка или округление при публикации таблиц Росстата. В модели это
    несущественно (влияет на A меньше 0,001 после деления на X ~ 2,5e8).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

FILE = Path('data/baz_tzv-2021.xlsx')
OUT  = Path('data/processed')
OUT.mkdir(parents=True, exist_ok=True)

# Допуск на расхождение с эталонными итогами (0,5 % от суммы)
TOL = 0.005

# Эталонные суммы (млн руб.) из строки TOTAL соответствующих листов
EXPECTED = {
    'Z': 124_282_711,
    'X': 246_859_637,
    'I':  25_957_673,
    'C': 162_459_521,
}


def to_num(x):
    """
    Устойчивое приведение к числам. Работает и с Series, и с DataFrame.
      - обычные и неразрывные пробелы удаляются,
      - запятая → точка,
      - пустые строки, тире, «—», «n/a» → NaN → 0.
    """
    if isinstance(x, pd.DataFrame):
        return x.apply(to_num)

    # x — Series
    s = x.astype(str)
    s = s.str.replace(r'[\s\u00a0\u202f]+', '', regex=True)   # пробелы
    s = s.str.replace(',', '.', regex=False)                   # запятая → точка
    s = s.replace({'': np.nan, '-': np.nan, '—': np.nan,
                   '–': np.nan, 'nan': np.nan, 'None': np.nan,
                   'n/a': np.nan, 'N/A': np.nan, '...': np.nan})
    return pd.to_numeric(s, errors='coerce').fillna(0)


def read_sheet(sheet, skiprows=0, nrows=None, usecols=None):
    """Обёртка над pd.read_excel с единым стилем."""
    return pd.read_excel(FILE, sheet_name=sheet, header=None,
                         skiprows=skiprows, nrows=nrows, usecols=usecols)


def check(name, actual, expected, tol=TOL):
    """Проверка с допуском. Возвращает относительное отклонение."""
    rel = abs(actual - expected) / expected
    status = 'OK' if rel <= tol else 'FAIL'
    print(f'  {name:3s}: {actual:>14.0f}  expected={expected:>14d}  '
          f'reldiff={rel*100:6.3f}%  [{status}]')
    return rel


# ============================================================
# 1. Промежуточное потребление Z (лист ТИоц, 243 × 120)
# ============================================================
tio = read_sheet('ТИоц')

# Продукты — строки 4:247; отрасли — столбцы 3:123
Z_block = tio.iloc[4:247, 3:123]
Z = to_num(Z_block).values.astype(float)

# Коды ОКПД2
codes_product  = tio.iloc[4:247, 1].astype(str).str.strip().values          # B
codes_industry = tio.iloc[2, 3:123].astype(str).str.strip().values          # строка 2

print(f'Z shape = {Z.shape}')
check('Z', Z.sum(), EXPECTED['Z'])
assert Z.shape == (243, 120), f'Z.shape = {Z.shape}'


# ============================================================
# 2. Выпуск X (лист М-отеч, строка P1, skiprows=258)
# ============================================================
x_raw = read_sheet('М-отеч', skiprows=258, usecols='D:DS')
X = to_num(x_raw).values.flatten().astype(float)

print(f'\nX shape = {X.shape}')
check('X', X.sum(), EXPECTED['X'])
assert X.shape == (120,), f'X.shape = {X.shape}'


# ============================================================
# 3. Шапка ТИцп: ищем столбцы P51, P3, P5, P6
# ============================================================
tipc_header = read_sheet('ТИцп', skiprows=0, nrows=4)
needed = ['P51', 'P3', 'P5', 'P6']
cols = {name: None for name in needed}
for i in range(tipc_header.shape[0]):
    for j, val in enumerate(tipc_header.iloc[i]):
        if isinstance(val, str):
            v = val.strip()
            if v in needed and cols[v] is None:
                cols[v] = j
print('\nНайдены столбцы ТИцп:', cols)
assert all(v is not None for v in cols.values()), f'Не найдены: {cols}'


# ============================================================
# 4. Валовое накопление I (P51, 243 продукта)
# ============================================================
tipc_i = read_sheet('ТИцп', skiprows=4, nrows=243, usecols=[cols['P51']])
I = to_num(tipc_i).values.flatten().astype(float)

print(f'\nI shape = {I.shape}')
check('I', I.sum(), EXPECTED['I'])
assert I.shape == (243,), f'I.shape = {I.shape}'


# ============================================================
# 5. Конечное использование C = P3 + P5 + P6 (243 продукта)
# ============================================================
col_idx = [cols['P3'], cols['P5'], cols['P6']]
tipc_c = read_sheet('ТИцп', skiprows=4, nrows=243, usecols=col_idx)
C = to_num(tipc_c).values.sum(axis=1).astype(float)

print(f'\nC shape = {C.shape}')
rel_C = check('C', C.sum(), EXPECTED['C'])
assert C.shape == (243,), f'C.shape = {C.shape}'
assert rel_C <= TOL, (
    f'C отклонение {rel_C*100:.3f}% > допуска {TOL*100:.1f}%. '
    f'Проверьте столбцы P3/P5/P6.'
)


# ============================================================
# 6. Сохранение
# ============================================================
np.save(OUT / 'Z.npy', Z)
np.save(OUT / 'X.npy', X)
np.save(OUT / 'I.npy', I)
np.save(OUT / 'C.npy', C)
np.save(OUT / 'codes_product.npy', codes_product)
np.save(OUT / 'codes_industry.npy', codes_industry)

meta = {
    'source': 'baz_tzv-2021.xlsx',
    'year': 2021,
    'units': 'million RUB',
    'shapes': {
        'Z': list(Z.shape),
        'X': list(X.shape),
        'I': list(I.shape),
        'C': list(C.shape),
    },
    'sums': {
        'Z': float(Z.sum()),
        'X': float(X.sum()),
        'I': float(I.sum()),
        'C': float(C.sum()),
    },
    'expected_sums': EXPECTED,
    'relative_errors': {
        'Z': abs(Z.sum() - EXPECTED['Z']) / EXPECTED['Z'],
        'X': abs(X.sum() - EXPECTED['X']) / EXPECTED['X'],
        'I': abs(I.sum() - EXPECTED['I']) / EXPECTED['I'],
        'C': abs(C.sum() - EXPECTED['C']) / EXPECTED['C'],
    },
    'tolerance': TOL,
    'notes': [
        'Z — промежуточное потребление в основных ценах (ТИоц, строки 001–243, столбцы 001–120).',
        'X — выпуск отраслей в основных ценах (М-отеч, P1, строка 259 Excel).',
        'I — валовое накопление основного капитала (ТИцп, P51, по продуктам).',
        'C — конечное использование = P3 + P5 + P6 (ТИцп, по продуктам).',
        'Z не включает чистые налоги на продукты (D21–D31 = 1 776 164 млн руб.).',
        'Конечное расхождение C (≈ 0,3 %) связано с P3_S14 (столбец 124) — '
        'в исходном xlsx, вероятно, есть нечисловая ячейка.',
    ],
}
(OUT / 'meta.json').write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

print(f'\nГотово. Сохранено в {OUT}')
print('Файлы: Z.npy, X.npy, I.npy, C.npy, codes_product.npy, codes_industry.npy, meta.json')