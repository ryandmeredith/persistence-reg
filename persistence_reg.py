from dataclasses import dataclass
from math import inf

from keras import Regularizer
from keras.ops import (
    append,
    argmax,
    array,
    cond,
    fori_loop,
    max,
    shape,
    slice_update,
    sum,
    where,
    zeros,
)


def _maximum_spanning_tree(weights):
    m, n = shape(weights)
    num_nodes = m + n

    max_weights = array((0.0,) + (-inf,) * (num_nodes - 1))
    visited = zeros(num_nodes, dtype=bool)

    initial_state = max_weights, visited

    def body(_, state):
        max_weights, visited = state
        i = argmax(where(visited, -inf, max_weights))

        new_visited = slice_update(visited, (i,), array((True,)))

        def input_node():
            potential_weights = weights[i, :]
            do_update = (potential_weights > max_weights[m:]) & ~visited[m:]
            update = where(do_update, potential_weights, max_weights[m:])
            return append(max_weights[:m], update)

        def output_node():
            potential_weights = weights[:, i - m]
            do_update = (potential_weights > max_weights[:m]) & ~visited[:m]
            update = where(do_update, potential_weights, max_weights[:m])
            return append(update, max_weights[m:])

        new_max_weights = cond(i < m, input_node, output_node)

        return new_max_weights, new_visited

    final_max_weights, _ = fori_loop(0, num_nodes, body, initial_state)

    return final_max_weights[1:]


@dataclass
class NeuralPersistence(Regularizer):
    norm: int = 2
    scale: float = 1.0

    def __call__(self, weights):
        absolute_weights = abs(weights)
        normalized = absolute_weights / max(absolute_weights)
        persistence = 1 - _maximum_spanning_tree(normalized)
        return -self.scale * sum(persistence**self.norm) ** (1 / self.norm)
