import copy
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate

from src.edge import Edge
from src.edge_group import EdgeGroup
from src.group_merger import GroupMerger


@dataclass
class CircuitComposer:
    """
    Class that composes quantum circuits implementing given continuous time quantum walks on arbitrary graphs.
    :var group_merger: GroupMerger object that decides how to group edges.
    """
    group_merger: GroupMerger

    @staticmethod
    def find_cx_arrangement(total_differ: int) -> list[ndarray]:
        """
        Finds cx gates necessary to straighten edges that span specified number of dimensions.
        :param total_differ: Total number of differing dimensions.
        :return: List of cx gates, specified as tuples of (control index, target index)
        """
        cx_gates = []
        fixed = np.zeros(total_differ, dtype=bool)
        differing_inds = np.where(~fixed)[0]
        while len(differing_inds) > 1:
            pairs = np.array(list(zip(differing_inds[::2], differing_inds[1::2])))
            cx_gates.extend(pairs)
            fixed[pairs[:, 1]] = True
            differing_inds = np.where(~fixed)[0]
        return cx_gates

    @staticmethod
    def apply_cx(edges: list[Edge], cx_gates: list[ndarray]) -> list[Edge]:
        """ Applies specified CX gates to specified edges and returns updated edges. """
        result = []
        for edge in edges:
            new_coords = np.copy(edge.coordinates)
            for gate in cx_gates:
                row = np.where(new_coords[:, gate[0]] == 1)[0][0]
                new_coords[row, gate[1]] = 1 - new_coords[row, gate[1]]
            result.append(Edge(new_coords))
        return result

    @staticmethod
    def compose_cx(cx_gates: list[ndarray], total_qubits: int) -> QuantumCircuit:
        """ Composes a circuit consisting of the specified CX gates and specified total number of qubits. """
        cx_qc = QuantumCircuit(total_qubits)
        for gate in cx_gates:
            cx_qc.cx(gate[0], gate[1])
        return cx_qc

    def compose_straight(self, edges: list[Edge], time: float) -> QuantumCircuit:
        """
        Composes a quantum circuit implementing a quantum walk, consisting of parallel straight edges only.
        :param edges: List of straight edges of a quantum walk on a hypercube that all span the same dimension.
        :param time: Walk time.
        :return: Quantum circuit implementing the given quantum walk.
        """
        edge_dim = np.where(edges[0].coordinates[0, :] != edges[0].coordinates[1, :])[0][0]
        edge_groups = [EdgeGroup.from_straight_edge(edge, i) for i, edge in enumerate(edges)]
        edge_groups = self.group_merger.merge_groups(edge_groups)

        qc = QuantumCircuit(edges[0].coordinates.shape[1])
        for group in edge_groups:
            rx_gate = RXGate(2 * time)
            control_inds = np.where(group.coordinates != -1)[0]
            if len(control_inds) > 0:
                control_state = ''.join([str(s) for s in group.coordinates[control_inds]])
                rx_gate = rx_gate.control(len(control_inds), ctrl_state=control_state)
            qc.append(rx_gate, control_inds.tolist() + [edge_dim])
        return qc

    def compose(self, edges: list[Edge], total_time: float, num_layers: int = 1) -> QuantumCircuit:
        """
        Composes a quantum circuit implementing an arbitrary quantum walk via trotterization over subgraphs.
        :param edges: List of edges of a quantum walk on a hypercube.
        :param total_time: Total walk time.
        :param num_layers: Number of trotterization layers. Higher - more accurate, but deeper circuit.
        :return: Quantum circuit implementing the given quantum walk.
        """
        edge_sets = defaultdict(list)
        for edge in edges:
            edge_sets[tuple(edge.differing_dimensions)].append(edge)

        layer_time = total_time / num_layers
        overall_qc = QuantumCircuit(edges[0].coordinates.shape[1])
        for edge_set in edge_sets.values():
            diff_dims = edge_set[0].differing_dimensions
            cx_gates = self.find_cx_arrangement(len(diff_dims))
            cx_gates = [diff_dims[gate] for gate in cx_gates]
            straight_edge_set = self.apply_cx(edge_set, cx_gates)

            cx_qc = self.compose_cx(cx_gates, overall_qc.num_qubits)
            edge_set_qc = self.compose_straight(straight_edge_set, layer_time)
            overall_qc = overall_qc.compose(cx_qc).compose(edge_set_qc).compose(cx_qc.inverse())

        for _ in range(num_layers - 1):
            overall_qc = overall_qc.compose(overall_qc)

        return overall_qc