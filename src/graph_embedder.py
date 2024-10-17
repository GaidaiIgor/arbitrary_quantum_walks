from abc import ABC, abstractmethod

from networkx import Graph


class GraphEmbedder(ABC):
    """ Represents a strategy for embedding graphs into hypercubes. """

    @abstractmethod
    def embed(self, graph: Graph, num_qubits: int) -> list[int]:
        """ Decides how to embed a given graph to a hypercube of bit strings corresponding to the given number of qubits.
        The edges of the hypercube connect nodes with a hamming distance of 1.
        Returns a list where i-th elem is an index of the hypercube node corresponding to i-th graph node. """
        pass


class GraphEmbedderTrivial(GraphEmbedder):
    def embed(self, graph: Graph, num_qubits: int) -> list[int]:
        """ Trivial identity embedding. """
        if len(graph) > 2 ** num_qubits:
            raise Exception('Embedding is impossible, graph is too large.')
        return list(range(len(graph)))
