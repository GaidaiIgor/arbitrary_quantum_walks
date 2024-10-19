""" Plots for the paper. """
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.general import Line, plot_general, save_figure


def plot_cx_count_vs_n():
    num_qubits = np.array(range(7, 11))
    num_edges = num_qubits - 1
    cx_exact = []
    cx_subgraph = []
    for n, m in zip(num_qubits, num_edges):
        df = pd.read_csv(f'../data/nodes_{n}/edges_{m}/out.csv')
        cx_exact.append(np.mean(df['exact_cx']))
        cx_subgraph.append(np.mean(df['subgraph_cx']))

    line_exact = Line(num_qubits, cx_exact, 0, label='exact')
    line_subgraph = Line(num_qubits, cx_subgraph, 1, label='subgraphs')
    plot_general([line_exact, line_subgraph], ('n', 'CX'), tick_multiples=(1, None))

    save_figure()


if __name__ == "__main__":
    plot_cx_count_vs_n()
    plt.show()
