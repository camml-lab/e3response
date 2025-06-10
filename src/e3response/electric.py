from collections.abc import Callable
from typing import Union

from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted
import tensorial.typing as tt

from . import keys

__all__ = "Polarization", "BornEffectiveCharges", "DielectricTensor"


def polarization(
    energy_fn: gcnn.GraphFunction,
    energy_field: str = atomic.TOTAL_ENERGY,
    efield_field: str = keys.EXTERNAL_ELECTRIC_FIELD,
    return_graph: bool = True,
) -> Callable:
    grad = gcnn.grad(
        of=f"globals.{energy_field}", wrt=f"globals.{efield_field}", sign=-1, has_aux=True
    )(energy_fn)

    def _calc(
        graph: jraph.GraphsTuple, evaluate_at: tt.ArrayType
    ) -> Union[tt.ArrayType, tuple[tt.ArrayType, jraph.GraphsTuple]]:
        res = grad(graph, evaluate_at)
        polarizations: jt.Float[tt.ArrayType, "g α"] = res[0]
        if return_graph:
            return polarizations, res[1]

        return polarizations

    return _calc


def dielectric_tensor(
    energy_fn: gcnn.GraphFunction,
    energy_field: str = atomic.TOTAL_ENERGY,
    external_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD,
    epsilon_0: float = 8.8541878128e-12,  # F/m
    include_identity: bool = True,
    return_graph: bool = True,
):
    """
    Returns a function that computes the (relative) dielectric tensor by differentiating
    the energy with respect to the applied electric field.

    This function first computes the polarization as the negative gradient of the energy
    with respect to the external electric field, and then computes the Jacobian of that
    polarization with respect to the electric field. This corresponds to the second derivative
    of the energy with respect to the field:

        ε_ij = (1 / ε₀) * ∂²E / ∂E_i ∂E_j

    If `include_identity=True`, the identity matrix is added to the resulting tensor,
    yielding the relative dielectric tensor (ε_r = χ + I), assuming linear response
    around vacuum.

    Parameters
    ----------
    energy_fn : gcnn.GraphFunction
        A function that computes the total energy of a system given a graph input.
    energy_field : str, optional
        The field name under which the total energy is stored in the graph.
        Default is `atomic.TOTAL_ENERGY`.
    external_field_key : str, optional
        The field name used for the applied external electric field in the graph.
        Default is `keys.EXTERNAL_ELECTRIC_FIELD`.
    epsilon_0 : float, optional
        Vacuum permittivity in F/m. Default is 8.8541878128e-12.
    include_identity : bool, optional
        If True, adds the identity matrix to the susceptibility to obtain the
        relative dielectric tensor. Default is True.
    return_graph : bool, optional
        If True, the returned function also returns the graph used during evaluation.
        Otherwise, only the dielectric tensor is returned. Default is True.

    Returns
    -------
    function
        A callable with signature:

            dielectric(graph: jraph.GraphsTuple, evaluate_at: Array)
                -> Array of shape (n_graph, 3, 3) or (Array, jraph.GraphsTuple)

        The output is the dielectric tensor for each graph in the batch, computed
        at the given external electric field.
    """
    calc_polarization = polarization(energy_fn, energy_field, external_field_key, return_graph=True)

    def shim(*args, **kwargs):
        # Create a shim that will sum over the batch before returning
        res = calc_polarization(*args, **kwargs)
        return res[0].sum(0), res[1]

    dielectric = jax.jacobian(shim, argnums=1, has_aux=True)

    def _calc(
        graph: jraph.GraphsTuple, evaluate_at: tt.ArrayType
    ) -> Union[tt.ArrayType, tuple[tt.ArrayType, jraph.GraphsTuple]]:
        res = dielectric(graph, evaluate_at)
        tensor: jt.Float[tt.ArrayType, "n_graph g α"] = res[0].swapaxes(0, 1)

        tensor = tensor / epsilon_0
        if include_identity:
            tensor += jnp.eye(3)

        if return_graph:
            return tensor, res[1]

        return tensor

    return _calc


class Polarization(linen.Module):
    """
    flax.linen.Module for computing the polarization vector
    from the total energy as a function of applied electric field.

    This module calculates the polarization vector :math:`\\mathbf{P}`
    using the first derivative of the total energy with respect to an applied
    homogeneous electric field.

    The polarization is defined as:

        .. math::

            P_\\alpha = - \\left. \\frac{\\partial E}{\\partial \\mathcal{E}_\\alpha}
                \\right|_{\\mathcal{E}=0}

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
    out_field = predicted(keys.POLARIZATION)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        # Define the gradient of the energy wrt electric field function
        self._grad_fn = gcnn.grad(
            of=f"globals.{self.energy_field}",
            wrt=f"globals.{keys.EXTERNAL_ELECTRIC_FIELD}",
            sign=-1,
            has_aux=True,
        )(self.energy_fn)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # res = self._grad_fn(graph, jnp.zeros_like(graph.globals[keys.EXTERNAL_ELECTRIC_FIELD]))
        res = self._grad_fn(graph, graph.globals[keys.EXTERNAL_ELECTRIC_FIELD])

        polarizations: jt.Float[tt.ArrayType, "g α"] = res[0]
        graph: jraph.GraphsTuple = res[1]
        updates = gcnn.utils.UpdateGraphDicts(graph)
        # updates.globals[self.out_field] = e3j.IrrepsArray("1o", polarizations)
        updates.globals[self.out_field] = polarizations

        return updates.get()


class DielectricTensor(linen.Module):
    """
    flax.linen.Module for computing the dielectric tensor.

    The dielectric tensor ε is defined as:

        ε_{αβ} = δ_{αβ} - (1/ε₀) * ∂²E / ∂𝔈_α ∂𝔈_β

    where:
        - E is the total energy,
        - 𝔈 is the applied electric field,
        - ε₀ is the vacuum permittivity.

    This module computes the susceptibility (∂P / ∂E), and optionally adds δ_{αβ}
    and rescales by ε₀ to get the dielectric tensor.

    Notes
    -----
    The calculation is done in the static, zero-field limit using autodiff.
    """

    energy_fn: gcnn.GraphFunction
    energy_field: str = predicted(atomic.TOTAL_ENERGY)
    external_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    epsilon_0: float = 8.8541878128e-12  # F/m
    include_identity: bool = True
    out_field: str = predicted(keys.DIELECTRIC_TENSOR)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._calc = dielectric_tensor(
            self.energy_fn,
            energy_field=self.energy_field,
            external_field_key=self.external_field_key,
            epsilon_0=self.epsilon_0,
            include_identity=self.include_identity,
            return_graph=True,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Evaluate the e-field derivative of the polarizability at zero electric field
        res = self._calc(graph, jnp.zeros_like(graph.globals[self.external_field_key]))
        dielectric: jt.Float[tt.ArrayType, "n_graph α β"] = res[0]
        graph = res[1]

        # Update the graph and return
        updates = gcnn.utils.UpdateGraphDicts(graph)
        updates.globals[self.out_field] = dielectric
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
        # pylint: disable=attribute-defined-outside-init
        # Compute the Jacobian of polarization with respect to atomic positions
        self._jacobian_fn = gcnn.jacobian(
            of=f"globals.{self.polarization_field}",
            wrt=f"nodes.{gcnn.keys.POSITIONS}",
            has_aux=True,
        )(self.polarization_fn)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        res = self._jacobian_fn(graph, graph.nodes[gcnn.keys.POSITIONS])
        born_tensors: jt.Float[tt.ArrayType, "κ α β"] = res[0].swapaxes(0, 1)
        graph: jraph.GraphsTuple = res[1]

        if gcnn.keys.CELL in graph.globals:
            n_atoms: int = graph.nodes[gcnn.keys.POSITIONS].shape[0]
            cells: jt.Float[tt.ArrayType, "n_graph 3 3"] = graph.globals[gcnn.keys.CELL]
            omega = jax.vmap(gcnn.calc.cell_volume)(cells)
            omega: jt.Float[tt.ArrayType, "κ"] = jnp.repeat(
                omega, graph.n_node, total_repeat_length=n_atoms
            )
            born_tensors = jax.vmap(jnp.multiply)(omega, born_tensors)

        updates = gcnn.utils.UpdateGraphDicts(graph)
        updates.nodes[predicted(keys.BORN_CHARGES)] = born_tensors
        return updates.get()
