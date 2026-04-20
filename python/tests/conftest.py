from pathlib import Path
import sys

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def X4(rng):
    """(50, 4) design matrix, no intercept."""
    return rng.standard_normal((50, 4))


@pytest.fixture
def y(rng):
    return rng.standard_normal(50)


@pytest.fixture
def Y3(rng):
    """(50, 3) outcome matrix."""
    return rng.standard_normal((50, 3))


@pytest.fixture
def binary_y(rng):
    return rng.integers(0, 2, size=50).astype(float)


@pytest.fixture
def count_y(rng):
    return rng.poisson(3.0, size=50).astype(float)
