import math

from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from tensorial import gcnn, nn_utils
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted
import tensorial.typing as tt

from . import keys

__all__ = "Polarization", "BornCharges", "DielectricTensor"

# Global quantities
ElectricFieldArray = jt.Float[tt.ArrayType, "num_graphs α"]
PolarizationArray = jt.Float[tt.ArrayType, "num_graphs α"]
DielectricTensorArray = jt.Float[tt.ArrayType, "num_graphs α b"]
# Per-atom quantities
PositionsArray = jt.Float[tt.ArrayType, "num_atoms γ"]
BornEffectiveChargesArray = jt.Float[tt.ArrayType, "num_atoms α γ"]
RamanTensorsArray = jt.Float[tt.ArrayType, "num_atoms γ α β"]


class Polarization(linen.Module):
    """
    Module for computing the macroscopic polarization vector as the negative derivative
    of the total energy with respect to an external electric field.

    In the context of linear response theory, the polarization **P** is defined as:

        P_α = -∂E / ∂ε_α

    where:
    - E is the total energy of the system,
    - ε_α is the α-component of the externally applied electric field,
    - The minus sign arises because the system lowers its energy in response to the applied field.

    This module evaluates the gradient of the total energy with respect to the electric field
    and returns the polarization as a global (graph-level) vector of shape `(3,)`.

    The calculation is done by automatic differentiation of `energy_fn` with respect to
    `globals.[electric_field_key]`. The resulting polarization vector is stored in the graph
    under `globals[out_key]`.

    Parameters
    ----------
    energy_fn : gcnn.GraphFunction
        A function that computes the total energy of the system from a graph input.
    energy_key : str, optional
        The field name under which the total energy is stored in the graph's globals.
        Defaults to `predicted(atomic.keys.ENERGY)`.
    electric_field_key : str, optional
        The field name under which the external electric field is stored in the graph's globals.
        Defaults to `keys.EXTERNAL_ELECTRIC_FIELD`.
    out_key : str, optional
        The field name under which to store the computed polarization in the graph's globals.
        Defaults to `predicted(keys.POLARIZATION)`.

    Returns
    -------
    jraph.GraphsTuple
        A graph with an additional field `globals[out_key]` storing the computed polarization
        vector of shape `(3,)` for each graph in the batch.
    """

    energy_fn: gcnn.GraphFunction
    energy_key: str = predicted(atomic.keys.ENERGY)
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    out_key: str = predicted(keys.POLARIZATION)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        # Define the gradient of the energy wrt electric field function
        self._calc = gcnn.experimental.diff(
            self.energy_fn,
            f"globals.{self.energy_key}:gk",
            wrt=[f"globals.{self.electric_field_key}:gα"],
            out=":gα",
            return_graph=True,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        derivative, graph = self._diff_fn(
            graph, jnp.zeros_like(graph.globals[keys.EXTERNAL_ELECTRIC_FIELD])
        )
        polarization: PolarizationArray = -derivative

        if gcnn.keys.CELL in graph.globals:
            # Normalize to get polarization per unit volume
            cells: jt.Float[tt.ArrayType, "n_graph 3 3"] = graph.globals[gcnn.keys.CELL]
            omega: jt.Float[tt.ArrayType, "n_graph"] = jax.vmap(gcnn.calc.cell_volume)(cells)
            # Mask off to avoid divide-by-zero
            graph_mask = nn_utils.prepare_mask(graph.globals.get(keys.MASK), omega)
            omega = jnp.where(graph_mask, omega, 1.0)
            polarization = jax.vmap(jnp.divide)(polarization, omega)

        return (
            gcnn.experimental.update_graph(graph).set(("globals", self.out_key), polarization).get()
        )


class DielectricTensor(linen.Module):
    """
    Returns a function that computes the (relative) dielectric tensor from the second
    derivative of the total energy with respect to the applied external electric field.

    The dielectric tensor ε_ij is related to how the energy of the system responds to
    perturbations in the electric field. This function computes:

        ε_ij = δ_ij - 4π * ∂²E / ∂E_i ∂E_j

    which is derived by differentiating the polarization vector:

        P_i = -∂E / ∂E_i

    and then evaluating the susceptibility:

        χ_ij = ∂P_i / ∂E_j = -∂²E / ∂E_i ∂E_j

    The returned function computes this susceptibility tensor, and—if `include_identity`
    is True—adds the identity matrix to yield the **relative dielectric tensor**:

        ε_r = I + 4π χ

    This assumes a linear response around vacuum permittivity (in Gaussian units).

    Parameters
    ----------
    energy_fn : gcnn.GraphFunction
        A function that computes the total energy of a system from a graph representation.
    energy_key : str, optional
        The key used to retrieve the total energy from the graph. Default is `atomic.TOTAL_ENERGY`.
    electric_field_key : str, optional
        The key under which the external electric field is stored in the graph globals.
        Default is `keys.EXTERNAL_ELECTRIC_FIELD`.
    include_identity : bool, optional
        If True, adds the identity tensor to the susceptibility to obtain the relative
        dielectric tensor. Default is True.
    return_graph : bool, optional
        If True, the returned function also returns the graph used for evaluation.
        Otherwise, only the dielectric tensor is returned. Default is True.

    Returns
    -------
    function
        A callable with signature:

            dielectric(graph: jraph.GraphsTuple, evaluate_at: Array)
                -> Array of shape (n_graph, 3, 3) or (Array, jraph.GraphsTuple)

        The output is the dielectric tensor for each graph in the batch, evaluated at
        the specified external electric field.
    """

    energy_fn: gcnn.GraphFunction
    energy_key: str = predicted(atomic.TOTAL_ENERGY)
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    include_identity: bool = True
    epsilon_0 = 1.0 / (4.0 * math.pi)  # If using atomic units
    out_key: str = predicted(keys.DIELECTRIC_TENSOR)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._calc = gcnn.experimental.diff(
            self.energy_fn,
            f"globals.{self.energy_key}:gk",
            wrt=[f"globals.{self.electric_field_key}:gα", f"globals.{self.electric_field_key}:gβ"],
            out=":gαβ",
            return_graph=True,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Evaluate the e-field derivative of the polarizability at zero electric field
        derivative, graph = self._calc(graph, graph.globals[self.electric_field_key])
        dielectric: DielectricTensorArray = -derivative / self.epsilon_0

        if gcnn.keys.CELL in graph.globals:
            cells: jt.Float[tt.ArrayType, "n_graph 3 3"] = graph.globals[gcnn.keys.CELL]
            omega: jt.Float[tt.ArrayType, "n_graph"] = jax.vmap(gcnn.calc.cell_volume)(cells)
            # Mask off to avoid divide-by-zero
            graph_mask = nn_utils.prepare_mask(graph.globals.get(keys.MASK), omega)
            omega = jnp.where(graph_mask, omega, 1.0)
            dielectric = jax.vmap(jnp.divide)(dielectric, omega)

        if self.include_identity:
            dielectric = jnp.eye(3) + dielectric

        # Update the graph and return
        return (
            gcnn.experimental.update_graph(graph).set(("globals", self.out_key), dielectric).get()
        )


class BornCharges(linen.Module):
    """
    Module for computing Born effective charge tensors using mixed second derivatives
    of the total energy with respect to atomic displacements and an applied electric field.

    The Born effective charge tensor describes how the macroscopic polarization of a material
    responds to atomic displacements in the presence of an electric field. It is defined as:

        Z*_{i,αγ} = -∂²E / ∂u_{iγ} ∂ε_α

    where:
    - E is the total energy of the system,
    - u_{iγ} is the displacement of atom i in Cartesian direction γ,
    - E_α is the α-component of the external electric field,
    - The negative sign arises because polarization is the negative derivative of energy
      with respect to the electric field.

    This module computes the Born effective charges per atom and stores them in the graph
    under the key `predicted(keys.BORN_CHARGES)` as a tensor of shape `(3, 3)` for each atom.

    Parameters
    ----------
    energy_fn : gcnn.GraphFunction
        A function that computes the total energy of the system from a graph input.
    energy_key : str, optional
        The field name under which the total energy is stored in the graph's globals.
        Defaults to `predicted(atomic.TOTAL_ENERGY)`.
    electric_field_key : str, optional
        The field name under which the external electric field is stored in the graph's globals.
        Defaults to `keys.EXTERNAL_ELECTRIC_FIELD`.

    Returns
    -------
    jraph.GraphsTuple
        A new graph with Born effective charge tensors attached to each node under
        `predicted(keys.BORN_CHARGES)`, with shape `(n_nodes, 3, 3)`.
    """

    energy_fn: gcnn.GraphFunction
    energy_key: str = predicted(atomic.TOTAL_ENERGY)
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    out_key: str = predicted(keys.BORN_CHARGES)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        # Define the derivative of the energy
        self._diff_fn = gcnn.experimental.diff(
            self.energy_fn,
            f"globals.{self.energy_key}:gk",
            wrt=[f"globals.{self.electric_field_key}:gα", f"nodes.{keys.POSITIONS}:Iγ"],
            out=":Iαγ",
            return_graph=True,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        derivative, graph = self._diff_fn(
            graph,
            graph.globals[keys.EXTERNAL_ELECTRIC_FIELD],
            graph.nodes[gcnn.keys.POSITIONS],
        )
        bec: BornEffectiveChargesArray = -derivative

        return gcnn.experimental.update_graph(graph).set(("nodes", self.out_key), bec).get()


class RamanTensors(linen.Module):
    """
    Module for computing Raman tensors via third derivatives of the total energy
    with respect to two electric field components and one atomic displacement.

    The Raman tensor quantifies the variation of the dielectric response (polarizability)
    under atomic displacements. In this implementation, the Raman tensor for each atom is
    computed as:

        R_{i,γαβ} = -∂³E / ∂u_{iγ} ∂ε_α ∂ε_β

    where:
    - E is the total energy of the system,
    - u_{iγ} is the displacement of atom i along Cartesian direction γ,
    - ε_α and ε_β are components of the applied electric field.

    The output is stored in the graph under the key `predicted(keys.RAMAN_TENSORS)` as a
    per-node (per-atom) tensor of shape `(3, 3, 3)`.

    Parameters
    ----------
    energy_fn : gcnn.GraphFunction
        A function that computes the total energy of the system from a graph input.
    energy_key : str, optional
        The field name under which the total energy is stored in the graph's globals.
        Defaults to `predicted(atomic.TOTAL_ENERGY)`.
    electric_field_key : str, optional
        The field name under which the external electric field is stored in the graph's globals.
        Defaults to `keys.EXTERNAL_ELECTRIC_FIELD`.

    Returns
    -------
    jraph.GraphsTuple
        A new graph with Raman tensors attached to each node under the key
        `predicted(keys.RAMAN_TENSORS)`.
    """

    energy_fn: gcnn.GraphFunction
    energy_key: str = predicted(atomic.TOTAL_ENERGY)
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    out_key: str = predicted(keys.RAMAN_TENSORS)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        # Define the derivative of the energy
        self._diff_fn = gcnn.experimental.diff(
            self.energy_fn,
            f"globals.{self.energy_key}:gk",
            wrt=[
                f"globals.{self.electric_field_key}:gα",
                f"globals.{self.electric_field_key}:gβ",
                f"nodes.{keys.POSITIONS}:Iγ",
            ],
            out=":Iγαβ",
            return_graph=True,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        derivative, graph = self._diff_fn(
            graph,
            graph.globals[keys.EXTERNAL_ELECTRIC_FIELD],
            graph.nodes[gcnn.keys.POSITIONS],
        )
        raman: RamanTensorsArray = -derivative

        return gcnn.experimental.update_graph(graph).set(("nodes", self.out_key), raman).get()
