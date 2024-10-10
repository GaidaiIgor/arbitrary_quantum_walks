import networkx as nx
import numpy as np
from networkx import Graph
from qiskit import transpile
from qiskit.quantum_info import Operator

from src.circuit_composer import CircuitComposer
from src.edge import Edge
from src.group_merger import GroupMerger


def graph_to_edge_list(graph: Graph) -> list[Edge]:
    """ Converts a given graph to a list of edges. """
    num_dims = round(np.log2(len(graph)))
    edges = []
    for edge in graph.edges:
        coordinates = []
        for node in edge:
            coordinates.append([int(c) for c in format(node, f'0{num_dims}b')])
        edges.append(Edge(np.array(coordinates)))
    return edges


def main():
    basis_gates = ['u3', 'cx']
    optimization_level = 3
    graphs = nx.read_graph6('data/graph4c.g6')
    group_merger = GroupMerger()
    circuit_composer = CircuitComposer(group_merger)
    for graph in graphs:
        edges = graph_to_edge_list(graph)
        qc = circuit_composer.compose(edges, 1, 1)
        op = Operator(qc).data
        qc_transpiled = transpile(qc, basis_gates=basis_gates, optimization_level=optimization_level)
        cx_count = qc_transpiled.count_ops().get('cx', 0)
        qc_depth = qc_transpiled.depth()


if __name__ == '__main__':
    main()
