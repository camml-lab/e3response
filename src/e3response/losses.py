from typing import Callable

import jax
import jraph

import tensorial.gcnn


def mlip_loss(
    energy: bool = True, forces: bool = True, energy_weight: float = 1.0, force_weight: float = 20.0
) -> Callable[[jraph.GraphsTuple, jraph.GraphsTuple], jax.Array]:
    weights: list[float] = []
    loss_terms = []

    if energy:
        weights.append(energy_weight)
        loss_terms.append(tensorial.gcnn.Loss("globals.predicted_energy", "globals.energy"))

    if forces:
        weights.append(force_weight)
        loss_terms.append(tensorial.gcnn.Loss("nodes.predicted_forces", "nodes.forces"))

    if not loss_terms:
        raise ValueError("Could not create loss function because all terms are set to `False`")

    if len(loss_terms):
        return tensorial.gcnn.WeightedLoss(weights, loss_terms)

    return loss_terms[0]
