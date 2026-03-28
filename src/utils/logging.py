"""
Shared logging setup for SLURM batch scripts.
"""

import logging
import os
import sys


def setup_slurm_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure unbuffered logging for SLURM environments.

    Sets PYTHONUNBUFFERED and configures logging to stderr with timestamps.
    Returns the root logger.
    """
    os.environ["PYTHONUNBUFFERED"] = "1"
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    return logging.getLogger()
