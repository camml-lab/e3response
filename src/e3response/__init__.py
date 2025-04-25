"""Library for machine learning on physical tensors"""

# flake8: noqa
# pylint: disable=wrong-import-position, wrong-import-order
import os

# Make sure we don't pre allocate memory, this is just antisocial
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import reax

from . import config, keys, metrics, stats

__version__ = "0.1.0"

__all__ = ("config", "metrics", "keys", "stats")


reax.metrics.get_registry().register_many(
    {
        "atomic/energy_per_atom_rmse": metrics.EnergyPerAtomRmse,
        "atomic/energy_per_atom_mae": metrics.EnergyPerAtomMae,
        "atomic/force_rmse": metrics.ForceRmse,
    }
)
