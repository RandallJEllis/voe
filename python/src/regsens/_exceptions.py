"""Custom exceptions and warnings for regsens."""


class RankDeficiencyWarning(UserWarning):
    """Issued when a column being added is nearly in the span of existing columns."""


class RankDeficiencyError(ValueError):
    """Raised when rank deficiency is detected and strict mode is enabled."""


class ConvergenceWarning(UserWarning):
    """Issued when IRLS did not converge within the maximum number of iterations."""
