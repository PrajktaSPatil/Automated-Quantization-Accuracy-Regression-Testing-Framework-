"""Project exceptions. Infrastructure failures must stay distinct from quality failures."""


class QuantGuardError(Exception):
    """Base error. The release gate maps this family to ERROR, never PASS."""


class ConfigurationError(QuantGuardError):
    """YAML is missing, malformed, or fails validation."""


class BaselineNotFoundError(QuantGuardError):
    """No approved baseline could be resolved from the configuration."""


class CandidateUnavailableError(QuantGuardError):
    """The candidate model or its artifact could not be loaded."""


class EvaluationError(QuantGuardError):
    """The evaluator crashed or returned an unusable result."""


class PolicyConfigError(QuantGuardError):
    """The policy document itself is invalid."""


class MetricMissingError(QuantGuardError):
    """A required metric was absent and the policy does not allow that."""
