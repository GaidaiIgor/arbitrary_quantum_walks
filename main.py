import os.path as path
import pickle
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import qiskit as qk
import scipy.linalg as lin
from networkx import Graph
from pandas import Series, DataFrame
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from tqdm import tqdm

from src.circuit_composer import CircuitComposer
from src.edge import Edge
from src.graph_embedder import GraphEmbedder, GraphEmbedderTrivial
from src.edge_merger import EdgeMerger


def generate_graphs():
    num_graphs = 10
    max_attempts = 10 ** 5
    nodes = 8
    edges = 7
    out_path = f'data/nodes_{nodes}/edges_{edges}'

    graphs = []
    disconnected_count = 0
    isomorphic_count = 0
    for i in range(max_attempts):
        next_graph = nx.gnm_random_graph(nodes, edges)
        connected = nx.is_connected(next_graph)
        isomorphic = any(nx.is_isomorphic(next_graph, g) for g in graphs)

        if not connected:
            disconnected_count += 1
        if isomorphic:
            isomorphic_count += 1
        if connected and not isomorphic:
            graphs.append(next_graph)
            print(f'\rGraphs generated: {len(graphs)}', end='')
        if len(graphs) == num_graphs:
            success = True
            break
    else:
        success = False

    print(f'\nTotal disconnected: {disconnected_count}')
    print(f'Total isomorphic: {isomorphic_count}')
    print(f'Success rate: {len(graphs) / i}')
    if success:
        print('Generation done')
    else:
        raise Exception('Failed to generate a valid graph set')

    with open(f'{out_path}/graphs.pkl', 'wb') as f:
        pickle.dump(graphs, f)


def graph_to_edge_list(graph: Graph, num_qubits: int, embedding: list[int]) -> list[Edge]:
    """
    Converts a given graph embedded in a hypercube according to embedding to a list of edges in the hypercube.
    :param graph: Graph to convert.
    :param num_qubits: Number of qubits for the hypercube.
    :param embedding: List of indices, where i-th element gives hypercube node index corresponding to i-th graph node.
    :return: List of edges on the hypercube.
    """
    assert len(embedding) == len(graph), 'Mismatching graph embedding'
    edges = []
    for edge in graph.edges:
        coordinates = []
        for node in edge:
            coordinates.append([int(c) for c in format(embedding[node], f'0{num_qubits}b')[::-1]])
        edges.append(Edge(np.array(coordinates)))
    return edges


def get_circuit_complexity(circuit: QuantumCircuit, basis_gates: list[str], optimization_level: int) -> tuple[int, int]:
    """ Transpiles given circuit and returns cx gate count and depth. """
    transpiled = qk.transpile(circuit, basis_gates=basis_gates, optimization_level=optimization_level)
    return transpiled.count_ops().get('cx', 0), transpiled.depth()


def process_row(df_row: tuple[int, Series], graphs: list[Graph], graph_embedder: GraphEmbedder, walk_time: float, circuit_composer: CircuitComposer, num_layers: int,
                basis_gates: list[str], optimization_level: int) -> Series:
    """
    Processes a given row in the dataframe and updates it with new information. In particular,
    1) creates an exact quantum circuit implementing walk on a given graph by direct exponentiation and decomposition.
    2) creates an approximate quantum circuit implementing walk on a given graph by trotterizing over subgraphs.
    :param df_row: A given row from dataframe to be modified with new properties.
    :param graphs: Overall list of graphs. Only the graph corresponding to the index of df_row will be processed.
    :param graph_embedder: Decides how to embed graph into hypercube.
    :param walk_time: Total walk time.
    :param circuit_composer: Composer class for the subgraph approach.
    :param num_layers: Number of trotterization layers for the subgraph approach.
    :param basis_gates: List of names of basis gates to transpile n-qubit operators to.
    :param optimization_level: Optimization level for transpilation.
    """
    index, series = df_row
    graph = graphs[index]

    adjacency_matrix = nx.adjacency_matrix(graph).toarray()
    num_qubits = round(np.ceil(np.log2(len(graph))))
    embedding = graph_embedder.embed(graph, num_qubits)
    hypercube_adjacency = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=int)
    hypercube_adjacency[np.ix_(embedding, embedding)] = adjacency_matrix

    exact_operator = Operator(lin.expm(-1j * walk_time * hypercube_adjacency))
    exact_qc = QuantumCircuit(num_qubits)
    exact_qc.append(exact_operator, list(range(num_qubits)))
    exact_cx, exact_depth = get_circuit_complexity(exact_qc, basis_gates, optimization_level)

    edges = graph_to_edge_list(graph, num_qubits, embedding)
    subgraph_qc = circuit_composer.compose(edges, walk_time, num_layers)
    subgraph_cx, subgraph_depth = get_circuit_complexity(subgraph_qc, basis_gates, optimization_level)
    subgraph_operator = Operator(subgraph_qc).data
    diff_norm = lin.norm(exact_operator - subgraph_operator, 2)
    print(diff_norm)

    series['exact_cx'] = exact_cx
    series['exact_depth'] = exact_depth
    series['subgraph_cx'] = subgraph_cx
    series['subgraph_depth'] = subgraph_depth
    series['diff_norm'] = diff_norm
    return series


def main():
    num_nodes = 8
    num_edges = 7
    walk_time = 0.1
    num_layers = 2
    basis_gates = ['rx', 'ry', 'rz', 'cx']
    optimization_level = 3
    num_workers = 1
    graph_embedder = GraphEmbedderTrivial()
    group_merger = EdgeMerger()
    circuit_composer = CircuitComposer(group_merger)
    process_func = partial(process_row, graph_embedder=graph_embedder, walk_time=walk_time, circuit_composer=circuit_composer, num_layers=num_layers, basis_gates=basis_gates,
                           optimization_level=optimization_level)
    data_path = f'data/nodes_{num_nodes}/edges_{num_edges}'

    with open(path.join(data_path, 'graphs.pkl'), 'rb') as f:
        graphs = pickle.load(f)
    process_func = partial(process_func, graphs=graphs)
    out_path = path.join(data_path, 'out.csv')
    df = pd.read_csv(out_path) if path.isfile(out_path) else DataFrame(index=range(len(graphs)))
    results = []
    if num_workers == 1:
        for result in tqdm(map(process_func, df.iterrows()), total=len(df), smoothing=0, ascii=' █'):
            results.append(result)
    else:
        with Pool(num_workers) as pool:
            for result in tqdm(pool.imap(process_func, df.iterrows()), total=len(df), smoothing=0, ascii=' █'):
                results.append(result)

    df = DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"Avg CX exact: {np.mean(df['exact_cx'])}\n")
    print(f"Avg depth exact: {np.mean(df['exact_depth'])}\n")
    print(f"Avg CX subgraph: {np.mean(df['subgraph_cx'])}\n")
    print(f"Avg depth subgraph: {np.mean(df['subgraph_depth'])}\n")
    print(f"Avg error: {np.mean(df['diff_norm'])}\n")


if __name__ == '__main__':
    np.set_printoptions(linewidth=250)
    main()
    # generate_graphs()
