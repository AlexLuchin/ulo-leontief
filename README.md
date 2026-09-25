# ulo-leontief

**Dynamic Leontief Model with Flexible Accelerator: The Case of Vehicle Recycling Fee in Russia**

**Динамическая модель Леонтьева с гибким акселератором: анализ последствий повышения утилизационного сбора в России**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22954178.svg)](https://doi.org/10.5281/zenodo.22954178)

---

## Abstract

**EN.** We develop a dynamic input–output Leontief model with a *flexible accelerator* to analyze the macroeconomic impact of the vehicle recycling fee (*utilizatsionny sbor*) hike in Russia. The classical Leontief dynamic model exhibits pathological instability: its transition matrix has all eigenvalues in the right half-plane. We propose a modification based on the flexible-accelerator hypothesis, where investment closes the gap between desired and actual capital, and on a *two-channel* representation of the recycling fee (a demand-side shock plus a targeted investment injection). The resulting system is stable for all scenario parameters, and the speed of convergence scales by a factor of 4 across three policy scenarios (growth / stagnation / crisis). Empirical calibration uses the 2021 Russian input–output tables published by Rosstat.

**RU.** Разработана динамическая модель межотраслевого баланса Леонтьева с *гибким акселератором* для анализа макроэкономических последствий повышения утилизационного сбора в России. Классическая динамическая модель Леонтьева обладает патологической неустойчивостью: все собственные значения матрицы динамики лежат в правой полуплоскости. Предложена модификация, основанная на гипотезе гибкого акселератора, где инвестиции закрывают разрыв между желаемым и фактическим капиталом, и на *двухканальном* представлении утилизационного сбора (шок спроса + адресная инвестиционная инъекция). Полученная система устойчива при всех сценарных параметрах, а скорость сходимости различается в 4 раза между тремя сценариями (рост / стагнация / падение). Эмпирическая калибровка — на базовых таблицах «затраты — выпуск» России за 2021 год (Росстат).

---

## Model / Модель

**State variables:** X(t) ∈ ℝ⁵ (output), K(t) ∈ ℝ⁵ (capital).

**Balance:**

    X = A·X + I_inv + I_rec + C

**Flexible accelerator:**

    I_inv(t) = α · (β · X(t) − K(t))

**Targeted recycling investment:**

    I_rec(t) = γ · ΔU · (1 − exp(−λ·t)) · e_rec

**Demand shock:**

    C(t) = C_0 − ΔU · e_1

After elimination of X the model reduces to a linear ODE in K:

    dK/dt = −M · K + b(t)

    M = α² · β · G + (α + δ) · I
    G = (I − A − α·β)⁻¹

**Stability:** min Re μ(M) > 0 for all scenarios.
**Critical bifurcation:** det(I − A − α*·β) = 0 ⇒ α* ≈ 1.25.

**Sectors (5):** automotive / metallurgy / fuel-energy / transport / other.

**Scenarios:**
- Growth:      α = 0.25, γ = 0.80
- Stagnation:  α = 0.10, γ = 0.30
- Crisis:      α = 0.02, γ = 0.00

---

## Installation / Установка

    git clone https://github.com/AlexLuchin/ulo-leontief.git
    cd ulo-leontief
    python -m venv .venv
    source .venv/bin/activate    # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

### Data / Данные

Download the 2021 Russian input–output tables from Rosstat:

    https://rosstat.gov.ru/folder/313/document/254127

Save the file as `data/baz_tzv-2021.xlsx` (≈ 1.8 MB).

The file is not included in the repository because of its size and to
respect Rosstat's distribution policy.

---

## Usage / Запуск

    # Full pipeline
    make run

    # Or manually
    python src/01_load.py
    python src/02_aggregate.py
    python src/03_matrices.py
    python src/04_scenarios.py
    python src/05_integrate.py
    python src/06_visualize.py

### Outputs / Результаты

| File | Description |
|---|---|
| `out/scen_X.png` | Output dynamics X(t)/X_0 for 5 sectors × 3 scenarios |
| `out/scen_K.png` | Capital dynamics K(t)/K_0 |
| `out/scen_bars.png` | Final output deviations by sector and scenario |
| `out/convergence.png` | ‖K(t) − K_∞‖ (semilog) |
| `out/adaptation_speed.png` | ‖X_auto(t) − X_∞‖ (semilog) |
| `out/adaptation_collapse.png` | Normalized deviation in scaled time t/τ |
| `out/bifurcation.png` | det(I − A − α·β) scan |

---

## Key results / Ключевые результаты

| Scenario | α | γ | τ (years) | ΔX_auto |
|---|---|---|---|---|
| Growth | 0.25 | 0.80 | **3.6** | −8.2 % |
| Stagnation | 0.10 | 0.30 | **7.3** | −8.1 % |
| Crisis | 0.02 | 0.00 | **14.2** | −8.0 % |

- Final output loss in automotive is **≈ 8 %** in all scenarios.
  The flexible accelerator amplifies the static shock of 5.5 % to ≈ 8 %
  through the capital-accumulation channel.
- Adaptation time differs **4×** between policies.
- The system is **stable** for all three scenarios — the flexible
  accelerator removes the classical Leontief pathology.

---

## Project structure / Структура проекта

    ulo-leontief/
    ├── src/
    │   ├── 01_load.py         # Read xlsx, extract Z, X, I, C
    │   ├── 02_aggregate.py    # 243×120 → 5×5 aggregation
    │   ├── 03_matrices.py     # Build A, β, δ, G, M, bifurcation scan
    │   ├── 04_scenarios.py    # Three scenarios (α, γ) + equilibrium
    │   ├── 05_integrate.py    # BDF integration of dK/dt = -M·K + b(t)
    │   └── 06_visualize.py    # Seven diagnostic plots
    ├── data/
    │   ├── baz_tzv-2021.xlsx  # (not in repo — download separately)
    │   └── processed/         # Generated .npy and .json
    ├── out/                   # Generated PNG plots
    ├── tests/                 # Unit and integration tests
    ├── requirements.txt
    ├── Makefile
    ├── LICENSE
    ├── CITATION.cff
    └── README.md

---

## Testing / Тестирование

    make test
    # or
    pytest tests/ -v

Tests are split in two groups:

- **Synthetic tests** (`tests/test_synthetic.py`) — generate a small
  3-sector model, verify invariants: A productive, M stable, bifurcation
  α* > 0, calibration residual < 1e-10. Run everywhere without external
  data.
- **Full tests** (`tests/test_full.py`) — verify sums of Z, X, I, C
  against published totals, check stability of M, check scenario
  ordering by τ. Skipped if `data/processed/*.npy` are absent.

---

## Citation / Цитирование

BibTeX:

    @software{luchin2026ulo,
      author    = {Luchin, Alex},
      title     = {ulo-leontief: Dynamic Leontief Model with
                   Flexible Accelerator},
      year      = {2026},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.22954178},
      url       = {https://github.com/AlexLuchin/ulo-leontief}
    }

See `CITATION.cff` for details.

---

## License / Лицензия

MIT — see [LICENSE](LICENSE).

---

## Author / Автор

**Alex Luchin** — ORCID [0009-0006-8972-0945](https://orcid.org/0009-0006-8972-0945)

---

## Acknowledgments / Благодарности

Input–output data: Federal State Statistics Service of the Russian
Federation (Rosstat), 2021 base tables, published 07.02.2025.

Numerical integration: SciPy `solve_ivp` with BDF method
(L-stable, suitable for stiff systems).
