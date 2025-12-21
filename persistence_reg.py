from dataclasses import asdict, dataclass
from math import inf

from keras import Regularizer
from keras.ops import (
    argmax,
    array,
    cond,
    fori_loop,
    full,
    max,
    maximum,
    shape,
    slice_update,
    sum,
    zeros,
)


def _maximum_spanning_tree(weights):
    m, n = shape(weights)
    num_nodes = m + n

    max_weights = full((num_nodes,), -inf)
    max_weights = slice_update(max_weights, (0,), array((0,)))
    visited_mask = zeros(num_nodes)

    initial_state = max_weights, visited_mask

    def body(_, state):
        max_weights, visited_mask = state
        i = argmax(max_weights + visited_mask)

        def input_node():
            update = maximum(weights[i, :] + visited_mask[m:], max_weights[m:])
            return slice_update(max_weights, (m,), update)

        def output_node():
            update = maximum(weights[:, i - m] + visited_mask[:m], max_weights[:m])
            return slice_update(max_weights, (0,), update)

        new_max_weights = cond(i < m, input_node, output_node)
        new_visited_mask = slice_update(visited_mask, (i,), array((-inf,)))

        return new_max_weights, new_visited_mask

    final_max_weights, _ = fori_loop(0, num_nodes, body, initial_state)

    return final_max_weights[1:]


@dataclass
class NeuralPersistence(Regularizer):
    scale: float = 1.0
    norm: int = 2

    def __call__(self, weights):
        absolute_weights = abs(weights)
        normalized = absolute_weights / max(absolute_weights)
        persistence = 1 - _maximum_spanning_tree(normalized)
        return -self.scale * sum(persistence**self.norm) ** (1 / self.norm)

    def get_config(self):
        asdict(self)
