from gudhi.simplex_tree import SimplexTree
from keras import Regularizer
from keras.ops import convert_to_numpy, max, sqrt
from numpy import arange, inf, mgrid, zeros


class NeuralPersistence(Regularizer):
    def __init__(self, in_dim, out_dim):
        self.in_nodes = in_dim
        self.out_nodes = out_dim
        self.total_nodes = in_dim + out_dim
        self.simplex = SimplexTree()

        self.simplex.insert_batch(
            arange(self.total_nodes)[None, :],
            zeros(self.total_nodes),
        )

    def __call__(self, weights):
        absolute_weights = abs(weights)
        normalized = 1 - absolute_weights / max(absolute_weights)
        self.simplex.insert_batch(
            mgrid[: self.in_nodes, self.in_nodes : self.total_nodes].reshape(2, -1),
            convert_to_numpy(normalized).reshape(-1),
        )
        try:
            self.simplex.compute_persistence()
            pairs = self.simplex.persistence_pairs()
        finally:
            self.simplex.reset_filtration(inf, min_dim=1)
        return sqrt(
            sum(
                normalized[min(edge), max(edge) - self.in_nodes] ** 2
                for _, edge in pairs
                if len(edge) == 2
            )
        )
