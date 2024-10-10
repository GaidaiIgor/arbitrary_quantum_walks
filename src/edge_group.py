from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from src.edge import Edge


@dataclass
class EdgeGroup:
    """
    A class representing a group of edges that can be implemented in 1 gate.
    :var coordinates: n coordinates of the group, -1 means that the corresponding dimension is spanned by the group.
    :var edge_inds: Indices from larger set of edges that this group covers.
    """
    coordinates: ndarray
    edge_inds: list[int]

    @staticmethod
    def from_straight_edge(edge: Edge, edge_ind: int) -> EdgeGroup:
        """
        Converts a single straight edge into a trivial EdgeGroup.
        :param edge: Edge to be converted.
        :param edge_ind: Index of the edge in a larger set.
        :return: Converted edge.
        """
        edge_group = EdgeGroup(edge.coordinates[0, :], [edge_ind])
        edge_dim = np.where(edge.coordinates[0, :] != edge.coordinates[1, :])[0][0]
        edge_group.coordinates[edge_dim] = -1
        return edge_group
