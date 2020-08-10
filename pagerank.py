import collections
import functools
import timeit
from typing import List, Tuple

import numpy as np
import jax.numpy as jnp
import jax


@functools.lru_cache()
def load_graph(N):
    """Load graph data created by make_graphs.py"""
    with open('%s.txt' % N) as f:
        f.readline()
        edge_data = []
        for line in f.readlines():
            x, y = line.split()
            edge_data.append((int(x), int(y)))
    return edge_data


@functools.lru_cache()
def adjacency_list(N) -> List[List[int]]:
    edge_data = load_graph(N)
    adjacencies = [[] for _ in range(N)]
    for from_node, to_node in edge_data:
        adjacencies[from_node].append(to_node)
    return adjacencies


@functools.lru_cache()
def adjacency_matrix(N) -> np.ndarray:
    edge_data = load_graph(N)
    adj_matrix = np.zeros([N, N], dtype=np.float32)
    for from_node, to_node in edge_data:
        adj_matrix[from_node, to_node] += 1
    return adj_matrix


@functools.lru_cache()
def flat_adjacency_list(N) -> Tuple[np.ndarray, np.ndarray]:
    edge_data = load_graph(N)
    from_nodes, to_nodes = zip(*edge_data)
    return np.array(from_nodes, dtype=np.int16), np.array(to_nodes, dtype=np.int16)


def pagerank_naive(N, num_iterations=100, d=0.85):
    node_data = [{'score': 1.0/N, 'new_score': 0} for _ in range(N)]
    adj_list = adjacency_list(N)

    for _ in range(num_iterations):
        for from_id, to_ids in enumerate(adj_list):
            score_packet = node_data[from_id]['score'] / len(to_ids)
            for to_id in to_ids:
                node_data[to_id]['new_score'] += score_packet
        for data_dict in node_data:
            data_dict['score'] = data_dict['new_score'] * d + (1 - d) / N
            data_dict['new_score'] = 0
    return np.array([data_dict['score'] for data_dict in node_data])


def pagerank_dense(N, num_iterations=100, d=0.85):
    adj_matrix = adjacency_matrix(N)
    transition_matrix = adj_matrix / np.sum(adj_matrix, axis=1, keepdims=True)
    transition_matrix = d * transition_matrix + (1 - d) / N

    score = np.ones([N], dtype=np.float32) / N
    for _ in range(num_iterations):
        score = score @ transition_matrix
    return score


def pagerank_sparse(N, num_iterations=100, d=0.85):
    from_nodes, to_nodes = flat_adjacency_list(N)
    neighbor_counts = np.array([len(l) for l in adjacency_list(N)], dtype=np.int16)

    score = np.ones([N], dtype=np.float32) / N
    for _ in range(num_iterations):
        score /= neighbor_counts
        score_packets = score[from_nodes]
        score = np.zeros([N], dtype=np.float32)
        np.add.at(score, to_nodes, score_packets)
        score = score * d + (1 - d) / N
    return score


def pagerank_sparse_bincount_trick(N, num_iterations=100, d=0.85):
    """Sparse implementation using bincount optimization.

    See https://github.com/numpy/numpy/issues/5922
    """
    from_nodes, to_nodes = flat_adjacency_list(N)
    neighbor_counts = np.array([len(l) for l in adjacency_list(N)], dtype=np.int16)

    score = np.ones([N], dtype=np.float32) / N
    for _ in range(num_iterations):
        score /= neighbor_counts
        score_packets = score[from_nodes]
        score = np.bincount(to_nodes, weights=score_packets, minlength=N)
        score = score * d + (1 - d) / N
    return score
    numpy.bincount(i, weights=a, minlength=1000)

def _jax_for_body_simple(N, d, score, from_nodes, to_nodes, neighbor_counts):
    score /= neighbor_counts
    score_packets = score[from_nodes]
    new_score = jax.ops.segment_sum(score_packets, to_nodes, num_segments=N)
    new_score = new_score * d + (1 - d) / N
    return new_score
_jax_for_body_simple = jax.jit(_jax_for_body_simple, static_argnums=(0,))


def pagerank_sparse_jax(N, num_iterations=100, d=0.85):
    from_nodes, to_nodes = flat_adjacency_list(N)
    neighbor_counts = np.array([len(l) for l in adjacency_list(N)], dtype=np.int16)
    score = np.ones([N], dtype=np.float32) / N

    for _ in range(num_iterations):
        score = _jax_for_body_simple(N, d, score, from_nodes, to_nodes, neighbor_counts)
    return score


def _jax_for_loop(num_iterations, N, d, score, from_nodes, to_nodes, neighbor_counts):
    def _jax_for_body_rolled(unused_i, val):
        score, = val
        score /= neighbor_counts
        score_packets = score[from_nodes]
        new_score = jax.ops.segment_sum(score_packets, to_nodes, num_segments=N)
        new_score = new_score * d + (1 - d) / N
        return (new_score,)
    init_val = (score,)
    return jax.lax.fori_loop(0, num_iterations, _jax_for_body_rolled, init_val)

_jax_for_loop = jax.jit(_jax_for_loop, static_argnums=(1,))


def pagerank_sparse_jax_rolled(N, num_iterations=100, d=0.85):
    from_nodes, to_nodes = flat_adjacency_list(N)
    neighbor_counts = np.array([len(l) for l in adjacency_list(N)], dtype=np.int16)
    score = np.ones([N], dtype=np.float32) / N

    score = _jax_for_loop(num_iterations, N, d, score, from_nodes, to_nodes, neighbor_counts)[0]
    return score


time_result = collections.namedtuple(
    'time_result', ['nodes', 'algorithm', 'time'])

def time_run(graph_size, algorithm, num_trials=10):
    # Prime the lru_caches and jax JIT so that all algorithms are on even footing
    algorithm(graph_size)
    time = timeit.timeit(lambda: algorithm(graph_size), number=num_trials) / num_trials
    return time_result(graph_size, algorithm.__name__, time)


def run_all(graph_sizes, algorithms):
    results = []
    for graph_size in graph_sizes:
        num_trials = min(100, max(3, 50000 // graph_size))
        for algo in algorithms:
            result = time_run(graph_size, algo, num_trials=num_trials)
            print(result, ',')
    # Copy data into Colab;  execute the following to generate plots.
    # Altair requires installing selenium or node.js to generate SVGs, so just use Colab...
"""
import collections
import altair as alt
import numpy as np
import pandas as pd
time_result = collections.namedtuple(
    'time_result', ['nodes', 'algorithm', 'time'])
data = pd.DataFrame(
## DATA GOES HERE
)
endpoints = np.array([10, 30000], dtype=np.float32)
endpoints3 = np.array([100, 10000], dtype=np.float32)
overlay_df = pd.concat([
    pd.DataFrame({'nodes': endpoints, 'time': 0.0001 * endpoints, 'scaling': 'linear'}),
    pd.DataFrame({'nodes': endpoints3, 'time': 0.0000000001 * endpoints3 ** 3, 'scaling': 'cubic'}),
])
dashed_lines = alt.Chart(overlay_df).mark_line().encode(
    x = alt.X('nodes', scale=alt.Scale(type='log', domain=(10, 30000))),
    y=alt.Y('time', scale=alt.Scale(type='log', domain=(1e-4, 1e2))),
    strokeDash=alt.StrokeDash('scaling', scale=alt.Scale(domain=['linear', 'cubic'], range=[[2, 2], [1, 4]])),
    color=alt.value('black'),
)

data_chart = alt.Chart(data2).mark_line().encode(
    x=alt.X('nodes', scale=alt.Scale(type='log', domain=(10, 30000))),
    y=alt.Y('time', scale=alt.Scale(type='log', domain=(1e-4, 1e2))),
    color='algorithm',
)

dashed_lines + data_chart
"""

if __name__ == '__main__':
    graph_sizes = (10, 30, 100, 300, 1000, 3000, 10000, 30000)
    algorithms = (
        pagerank_naive,
        pagerank_sparse,
        pagerank_sparse_bincount_trick,
        pagerank_dense,
        pagerank_sparse_jax,
        pagerank_sparse_jax_rolled,
    )
    run_all(graph_sizes, algorithms)
