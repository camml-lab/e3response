from flax import linen
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic, keys


class LennardJonesEnergy(linen.Module):
    """Simple Lennard-Jones potential"""

    epsilon: float = 1.0
    sigma: float = 1.0

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        graph = gcnn.with_edge_vectors(graph)
        energy = (self.sigma / graph.edges[gcnn.keys.EDGE_LENGTHS]) ** 6.0
        energy = 2.0 * self.epsilon * (energy**2 - energy)

        nodes = graph.nodes
        # Now, sum the edge energies onto each atom
        per_atom_energy = jraph.segment_sum(
            energy, graph.receivers, num_segments=len(graph.nodes[keys.POSITIONS])
        )

        if atomic.ENERGY_PER_ATOM in nodes:
            per_atom_energy = per_atom_energy + nodes[atomic.ENERGY_PER_ATOM]
        nodes[atomic.ENERGY_PER_ATOM] = per_atom_energy

        return graph._replace(nodes=nodes)
