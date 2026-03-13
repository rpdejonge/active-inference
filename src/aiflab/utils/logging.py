"""
Logging configuration.
"""

import logging


def configure_logging(level: str = "INFO") -> None:
    """
    Configure basic logging for the application.
    """

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )