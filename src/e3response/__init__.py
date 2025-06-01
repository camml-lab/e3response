"""Library for machine learning on physical tensors"""

# flake8: noqa
# pylint: disable=wrong-import-position, wrong-import-order
import os

# Make sure we don't pre allocate memory, this is just antisocial
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import reax

from . import config, keys

__version__ = "0.1.0"

__all__ = (
    "config",
    "keys",
)
