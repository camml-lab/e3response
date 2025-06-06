from collections.abc import Callable
from typing import Union

import jax
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted

from . import keys


def response_loss(
    energy: Union[bool, float] = True,
    forces: Union[bool, float] = True,
    born_charges: Union[bool, float] = True,
) -> Callable[[jraph.GraphsTuple, jraph.GraphsTuple], jax.Array]:
    weights: list[float] = []
    loss_terms = []

    if energy:
        weights.append(energy if isinstance(energy, bool) else 1.0)
        loss_terms.append(
            gcnn.Loss(f"globals.{predicted(atomic.TOTAL_ENERGY)}", f"globals.{atomic.TOTAL_ENERGY}")
        )

    if forces:
        weights.append(forces if isinstance(forces, bool) else 1.0)
        loss_terms.append(gcnn.Loss(f"nodes.{predicted(atomic.FORCES)}", f"nodes.{atomic.FORCES}"))

    if born_charges:
        weights.append(born_charges if isinstance(born_charges, bool) else 1.0)
        loss_terms.append(
            gcnn.Loss(f"nodes.{predicted(keys.BORN_CHARGES)}", f"nodes.{keys.BORN_CHARGES}")
        )

    if not loss_terms:
        raise ValueError(
            "Could not create loss function because all terms (energy, forces, ...) are set to `False`"
        )

    if len(loss_terms):
        return gcnn.WeightedLoss(weights, loss_terms)

    return loss_terms[0]
