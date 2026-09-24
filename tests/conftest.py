"""
Shared pytest fixtures for ulo-leontief tests.

Две группы тестов:
  - synthetic: работают на маленькой 3-секторной модели, не требуют данных
  - full:      проверяют суммы Z, X, I, C из базовых таблиц Росстата
               (пропускаются, если data/processed/*.npy отсутствуют)
"""
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
OUT  = ROOT / 'out'


def _have(name):
    """Проверяет, что файл существует в data/processed."""
    return (DATA / name).exists()


@pytest.fixture
def processed_dir():
    """Путь к папке с обработанными данными."""
    return DATA


@pytest.fixture
def have_full_data():
    """True, если все ключевые .npy на месте."""
    required = ['Z.npy', 'X.npy', 'I.npy', 'C.npy',
                'A.npy', 'B.npy', 'M.npy']
    return all(_have(n) for n in required)


@pytest.fixture
def synthetic_model():
    """
    Маленькая синтетическая 3-секторная модель с известными свойствами.

    Используется для тестов инвариантов, которые не требуют внешних данных:
      - A продуктивна (суммы по столбцам < 1),
      - M устойчива (min Re μ > 0),
      - калибровка X = G(C0 - α K0) воспроизводит X5 с точностью 1e-10,
      - бифуркация α* > 0.
    """
    A = np.array([
        [0.10, 0.02, 0.01],
        [0.15, 0.20, 0.03],
        [0.05, 0.06, 0.08],
    ])
    beta  = np.array([0.5, 0.4, 0.3])
    delta = 0.05
    alpha = 0.10
    B_eff = np.diag(beta)

    G = np.linalg.inv(np.eye(3) - A - alpha * B_eff)
    M = alpha**2 * B_eff @ G + (alpha + delta) * np.eye(3)

    return dict(
        A=A, beta=beta, delta=delta, alpha=alpha,
        B_eff=B_eff, G=G, M=M, n=3,
    )