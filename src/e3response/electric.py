import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted
import tensorial.typing

from . import keys

__all__ = "Polarization", "BornEffectiveCharges"


class Polarization(linen.Module):
    """
    flax.linen.Module for computing the polarization vector
    from the total energy as a function of applied electric field.

    This module calculates the polarization vector :math:`\\mathbf{P}`
    using the first derivative of the total energy with respect to an applied
    homogeneous electric field.

    The polarization is defined as:

        .. math::

            P_\\alpha = - \\left. \\frac{\\partial E}{\\partial \\mathcal{E}_\\alpha} \\right|_{\\mathcal{E}=0}

    where:
        - :math:`E` is the total energy of the system,
        - :math:`\\mathcal{E}` is the applied electric field,
        - :math:`\\alpha` ∈ {x, y, z} is a Cartesian direction.

    Notes
    -----
    The polarization describes the dipole moment per unit volume induced by an external field
    in the system. This definition corresponds to the *modern theory of polarization* in the
    context of finite systems or within a linear response framework for periodic systems.

    This implementation assumes that the energy is differentiable with respect to the electric
    field, and that the polarization is computed in the static (zero-frequency) limit.
    """

    energy_fn: gcnn.GraphFunction
    energy_field = predicted(atomic.keys.ENERGY)

    def setup(self) -> None:
        # Define the gradient of the energy wrt electric field function
        self._grad_fn = gcnn.grad(
            of=f"globals.{self.energy_field}",
            wrt=f"globals.{keys.EXTERNAL_ELECTRIC_FIELD}",
            sign=-1,
            out_field=None,
        )(self.energy_fn)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        polarizations: jt.Float[tensorial.typing.ArrayType, "g α"] = self._grad_fn(
            graph, jnp.zeros_like(graph.globals[keys.EXTERNAL_ELECTRIC_FIELD])
        )
        updates = gcnn.utils.UpdateGraphDicts(graph)
        updates.globals[f"predicted_{keys.POLARIZATION}"] = e3j.IrrepsArray("1o", polarizations)
        # updates.globals[predicted(keys.POLARIZATION)] = polarizations

        return updates.get()


class BornEffectiveCharges(linen.Module):
    """
    flax.linen.Module for computing Born effective charge tensors.

    The Born effective charge tensor Z* for atom κ is defined as:

        Z^*_{κ, αβ} = Ω * ∂P_α / ∂u_{κβ}

    where:
        - P_α is the α component of the polarization vector,
        - u_{κβ} is the β component of the displacement of atom κ,
        - Ω is the volume of the unit cell (optional, may be included externally).

    This module computes Z* as the gradient of polarization with respect to atomic positions.
    """

    polarization_fn: gcnn.GraphFunction
    polarization_field: str = predicted(keys.POLARIZATION)

    def setup(self) -> None:
        # Compute the Jacobian of polarization with respect to atomic positions
        self._jacobian_fn = gcnn.jacobian(
            of=f"globals.{self.polarization_field}",
            wrt=f"nodes.{gcnn.keys.POSITIONS}",
            out_field=None,
        )(self.polarization_fn)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        born_tensors: jt.Float[tensorial.typing.ArrayType, "κ α β"] = self._jacobian_fn(
            graph, graph.nodes[gcnn.keys.POSITIONS]
        ).swapaxes(0, 1)
        if gcnn.keys.CELL in graph.globals:
            cells = graph.globals[gcnn.keys.CELL]
            omega = jax.vmap(gcnn.calc.cell_volume)(cells)
            born_tensors = jnp.repeat(omega, graph.n_node) * born_tensors

        updates = gcnn.utils.UpdateGraphDicts(graph)
        updates.nodes[predicted(keys.BORN_CHARGES)] = born_tensors
        return updates.get()
