"""regsens — efficient regression sensitivity analysis via QR updating."""

from .linear import LinearSensitivity
from .glm import GLMSensitivity
from .families import Gaussian, Binomial, Poisson, get_family, FAMILIES
from .results import SensitivityResult
from ._exceptions import RankDeficiencyWarning, RankDeficiencyError, ConvergenceWarning

__version__ = "0.1.0"
__all__ = [
    "LinearSensitivity",
    "GLMSensitivity",
    "Gaussian",
    "Binomial",
    "Poisson",
    "get_family",
    "FAMILIES",
    "SensitivityResult",
    "RankDeficiencyWarning",
    "RankDeficiencyError",
    "ConvergenceWarning",
]
