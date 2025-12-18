from math import inf
from dataclasses import dataclass

from keras import Regularizer
from keras.ops import (
    argmin,
    array,
    cond,
    fori_loop,
    full,
    max,
    nonzero,
    scatter_update,
    shape,
    slice_update,
    where,
    zeros,
)


def _minimal_spanning_tree(weights, start_index=0):
    m, n = shape(weights)
    num_nodes = m + n

    min_weights = full((num_nodes,), inf)
    min_weights = slice_update(min_weights, array((start_index,)), array((0.0,)))
    visited = zeros(num_nodes, dtype=bool)
    total = array(0.0)

    initial_state = min_weights, visited, total

    def body(_, state):
        min_weights, visited, total = state
        i = argmin(where(visited, inf, min_weights))

        new_total = total + min_weights[i]
        new_visited = slice_update(visited, array((i,)), array((True,)))

        def input_node():
            potential_weights = weights[i, :]
            mask = (potential_weights < min_weights[m:]) & (~visited[m:])
            return scatter_update(
                min_weights, nonzero(mask).T + m, potential_weights[mask]
            )

        def output_node():
            potential_weights = weights[:, i - m]
            mask = (potential_weights < min_weights[:m]) & (~visited[:m])
            return scatter_update(min_weights, nonzero(mask).T, potential_weights[mask])

        new_min_weights = cond(i < m, input_node, output_node)

        return new_min_weights, new_visited, new_total

    _, _, total = fori_loop(0, num_nodes, body, initial_state)

    return total


@dataclass
class NeuralPersistence(Regularizer):
    norm: int = 2
    scale: float = 1.0

    def __call__(self, weights):
        absolute_weights = abs(weights)
        normalized = (1 - absolute_weights / max(absolute_weights)) ** self.norm
        return -self.scale * _minimal_spanning_tree(normalized) ** (1 / self.norm)
