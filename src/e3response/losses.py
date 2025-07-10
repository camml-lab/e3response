from collections.abc import Callable

import jax
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted

from . import keys


def response_loss(
    energy: bool | float = False,
    forces: bool | float = False,
    polarization_tensors: bool | float = False,
    dielectric_tensor: bool | float = False,
    born_charges: bool | float = False,
    raman_tensors: bool | float = False,
) -> Callable[[jraph.GraphsTuple, jraph.GraphsTuple], jax.Array]:
    weights: list[float] = []
    loss_terms = []

    if energy:
        weights.append(1.0 if isinstance(energy, bool) else energy)
        loss_terms.append(
            gcnn.Loss(f"globals.{predicted(atomic.TOTAL_ENERGY)}", f"globals.{atomic.TOTAL_ENERGY}")
        )

    if forces:
        weights.append(1.0 if isinstance(forces, bool) else forces)
        loss_terms.append(gcnn.Loss(f"nodes.{predicted(atomic.FORCES)}", f"nodes.{atomic.FORCES}"))

    if born_charges:
        weights.append(1.0 if isinstance(born_charges, bool) else born_charges)
        loss_terms.append(
            gcnn.Loss(f"nodes.{predicted(keys.BORN_CHARGES)}", f"nodes.{keys.BORN_CHARGES}")
        )

    if polarization_tensors:
        weights.append(1.0 if isinstance(polarization_tensors, bool) else polarization_tensors)
        loss_terms.append(
            gcnn.Loss(f"globals.{predicted(keys.POLARIZATION)}", f"globals.{keys.POLARIZATION}")
        )

    if dielectric_tensor:
        weights.append(1.0 if isinstance(dielectric_tensor, bool) else dielectric_tensor)
        loss_terms.append(
            gcnn.Loss(
                f"globals.{predicted(keys.DIELECTRIC_TENSOR)}", f"globals.{keys.DIELECTRIC_TENSOR}"
            )
        )

    if raman_tensors:
        weights.append(1.0 if isinstance(raman_tensors, bool) else raman_tensors)
        loss_terms.append(
            gcnn.Loss(f"nodes.{predicted(keys.RAMAN_TENSORS)}", f"nodes.{keys.RAMAN_TENSORS}")
        )

    if not loss_terms:
        raise ValueError(
            "Could not create loss function because all terms (energy, forces, ...) are set to "
            "`False`"
        )

    if loss_terms:
        return gcnn.WeightedLoss(loss_terms, weights)

    return loss_terms[0]
