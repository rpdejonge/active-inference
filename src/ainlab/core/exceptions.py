"""
Project-specific exceptions.
"""


class AINLabError(Exception):
    """Base exception for the project."""


class ConfigurationError(AINLabError):
    """Raised when configuration is invalid."""


class RegistryError(AINLabError):
    """Raised when registry lookups fail."""
