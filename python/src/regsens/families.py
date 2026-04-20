"""
GLM family objects: link, inverse-link, variance, IRLS weights, working response.

Each family is picklable (module-level class) and serialisable via to_dict() / from_dict()
for cross-process transmission through ProcessPoolExecutor.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class Family(ABC):
    """Abstract GLM family."""

    name: str

    @abstractmethod
    def link(self, mu: np.ndarray) -> np.ndarray:
        """g(μ) — maps mean to linear predictor."""

    @abstractmethod
    def inv_link(self, eta: np.ndarray) -> np.ndarray:
        """g^{-1}(η) — maps linear predictor to mean."""

    @abstractmethod
    def link_deriv(self, mu: np.ndarray) -> np.ndarray:
        """g'(μ) = dη/dμ."""

    @abstractmethod
    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(μ) — variance function."""

    def irls_weights(self, mu: np.ndarray) -> np.ndarray:
        """w_i = 1 / (V(μ_i) · g'(μ_i)²), clamped to [1e-10, 1e10]."""
        w = 1.0 / (self.variance(mu) * self.link_deriv(mu) ** 2)
        return np.clip(w, 1e-10, 1e10)

    def working_response(self, y: np.ndarray, mu: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """z_i = η_i + (y_i - μ_i) · g'(μ_i)."""
        return eta + (y - mu) * self.link_deriv(mu)

    @abstractmethod
    def init_mu(self, y: np.ndarray) -> np.ndarray:
        """Safe starting μ for IRLS."""

    @abstractmethod
    def deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Total deviance = −2 log-likelihood (up to additive constant)."""

    def null_deviance(self, y: np.ndarray) -> float:
        """Deviance of the intercept-only model."""
        mu_null = np.full(len(y), float(np.mean(self.init_mu(y))))
        return self.deviance(y, mu_null)

    def to_dict(self) -> dict:
        """Serialise for cross-process pickling."""
        return {"name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> Family:
        return FAMILIES[d["name"]]()


class Gaussian(Family):
    """Gaussian family with identity link."""

    name = "gaussian"

    def link(self, mu):       return mu.copy()
    def inv_link(self, eta):  return eta.copy()
    def link_deriv(self, mu): return np.ones_like(mu)
    def variance(self, mu):   return np.ones_like(mu)
    def irls_weights(self, mu): return np.ones(len(mu))
    def init_mu(self, y):     return y.copy()

    def deviance(self, y, mu):
        return float(np.sum((y - mu) ** 2))

    def null_deviance(self, y):
        return float(np.sum((y - y.mean()) ** 2))


class Binomial(Family):
    """Binomial family with logit link."""

    name = "binomial"
    _EPS = 1e-8

    def link(self, mu):
        mu_c = np.clip(mu, self._EPS, 1 - self._EPS)
        return np.log(mu_c / (1 - mu_c))

    def inv_link(self, eta):
        # Numerically stable sigmoid avoiding overflow
        return np.where(
            eta >= 0,
            1.0 / (1.0 + np.exp(-eta)),
            np.exp(eta) / (1.0 + np.exp(eta)),
        )

    def link_deriv(self, mu):
        mu_c = np.clip(mu, self._EPS, 1 - self._EPS)
        return 1.0 / (mu_c * (1 - mu_c))

    def variance(self, mu):
        mu_c = np.clip(mu, self._EPS, 1 - self._EPS)
        return mu_c * (1 - mu_c)

    def irls_weights(self, mu):
        # For logit: w = mu*(1-mu) directly (cancels link_deriv squared)
        return np.clip(self.variance(mu), 1e-10, 1e10)

    def working_response(self, y, mu, eta):
        mu_c = np.clip(mu, self._EPS, 1 - self._EPS)
        w = mu_c * (1 - mu_c)
        return eta + (y - mu_c) / np.where(w < 1e-10, 1e-10, w)

    def init_mu(self, y):
        return np.clip((y + 0.5) / 2, self._EPS, 1 - self._EPS)

    def deviance(self, y, mu):
        mu_c = np.clip(mu, self._EPS, 1 - self._EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(y > 0, y * np.log(y / mu_c), 0.0)
            t2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu_c)), 0.0)
        return float(2 * np.sum(t1 + t2))

    def null_deviance(self, y):
        p_bar = np.clip(y.mean(), self._EPS, 1 - self._EPS)
        return self.deviance(y, np.full(len(y), p_bar))


class Poisson(Family):
    """Poisson family with log link."""

    name = "poisson"
    _EPS = 1e-8

    def link(self, mu):       return np.log(np.maximum(mu, self._EPS))
    def inv_link(self, eta):  return np.exp(np.clip(eta, -500, 500))
    def link_deriv(self, mu): return 1.0 / np.maximum(mu, self._EPS)
    def variance(self, mu):   return np.maximum(mu, self._EPS)

    def irls_weights(self, mu):
        # For log link: w = mu directly
        return np.maximum(mu, 1e-10)

    def working_response(self, y, mu, eta):
        mu_s = np.maximum(mu, self._EPS)
        return eta + (y - mu_s) / mu_s

    def init_mu(self, y):
        return np.maximum(y, 0.1)

    def deviance(self, y, mu):
        mu_c = np.maximum(mu, self._EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y > 0, y * np.log(y / mu_c), 0.0) - (y - mu_c)
        return float(2 * np.sum(t))

    def null_deviance(self, y):
        mu_null = np.full(len(y), max(float(y.mean()), self._EPS))
        return self.deviance(y, mu_null)


FAMILIES: dict[str, type[Family]] = {
    "gaussian": Gaussian,
    "binomial": Binomial,
    "poisson": Poisson,
}


def get_family(spec) -> Family:
    """Resolve a family spec (str or Family instance) to a Family object."""
    if isinstance(spec, Family):
        return spec
    if isinstance(spec, str):
        key = spec.lower()
        if key not in FAMILIES:
            raise ValueError(f"Unknown family {spec!r}. Choose from {list(FAMILIES)}")
        return FAMILIES[key]()
    raise TypeError(f"family must be str or Family instance, got {type(spec)}")
