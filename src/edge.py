from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray


@dataclass
class Edge:
    """
    Class representing an edge in a walk on a hypercube.
    :var coordinates: 2 x n array of hypercube coordinates for the 2 endpoints of the edge
    """
    coordinates: ndarray
    differing_dimensions: ndarray = field(init=False)

    def __post_init__(self):
        self.differing_dimensions = np.where(self.coordinates[0, :] != self.coordinates[1, :])[0]
