import collections
import functools
import timeit
from typing import List, Tuple

import numba
import numpy as np
import jax.numpy as jnp
import jax
import tensorflow as tf


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
    return (np.array(from_nodes, dtype=np.int16),
            np.array(to_nodes, dtype=np.int16))


@functools.lru_cache()
def get_neighbor_count(N):
    return np.array([len(l) for l in adjacency_list(N)], dtype=np.int16)


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


def pagerank_dense_np(N, num_iterations=100, d=0.85):
    adj_matrix = adjacency_matrix(N)
    transition_matrix = adj_matrix / np.sum(adj_matrix, axis=1, keepdims=True)
    transition_matrix = d * transition_matrix + (1 - d) / N

    score = np.ones([N], dtype=np.float32) / N
    for _ in range(num_iterations):
        score = score @ transition_matrix
    return score


def pagerank_sparse_np(N, num_iterations=100, d=0.85):
    from_nodes, to_nodes = flat_adjacency_list(N)
    neighbor_counts = get_neighbor_count(N)

    score = np.ones([N], dtype=np.float32) / N
    for _ in range(num_iterations):
        score /= neighbor_counts
        score_packets = score[from_nodes]
        score = np.zeros([N], dtype=np.float32)
        np.add.at(score, to_nodes, score_packets)
        score = score * d + (1 - d) / N
    return score


def pagerank_sparse_np_bincount(N, num_iterations=100, d=0.85):
    """Sparse implementation using bincount optimization.

    See https://github.com/numpy/numpy/issues/5922
    """
    from_nodes, to_nodes = flat_adjacency_list(N)
    neighbor_counts = get_neighbor_count(N)

    score = np.ones([N], dtype=np.float32) / N
    for _ in range(num_iterations):
        score /= neighbor_counts
        score_packets = score[from_nodes]
        score = np.bincount(to_nodes, weights=score_packets, minlength=N)
        score = score * d + (1 - d) / N
    return score
    numpy.bincount(i, weights=a, minlength=1000)


@numba.jit(nopython=True)
def _numba_for_loop(num_iterations, N, d, from_nodes, to_nodes, neighbor_counts):
    score = np.ones(N, dtype=np.float32) / N
    for _ in range(num_iterations):
        score /= neighbor_counts
        new_score = np.zeros_like(score)
        for i in range(from_nodes.shape[0]):
            new_score[to_nodes[i]] += score[from_nodes[i]]
        score = new_score
        score = score * d + (np.float32(1) - d) / np.float32(N)
    return score


def pagerank_sparse_numba(N, num_iterations=100, d=0.85):
    from_nodes, to_nodes = flat_adjacency_list(N)
    neighbor_counts = get_neighbor_count(N)
    # Numba assumes float literals are float64, causing typecasting havoc.
    d = np.float32(d)
    return _numba_for_loop(num_iterations, N, d, from_nodes, to_nodes, neighbor_counts)


def _jax_for_body_simple(N, d, score, from_nodes, to_nodes, neighbor_counts):
    score /= neighbor_counts
    score_packets = score[from_nodes]
    new_score = jax.ops.segment_sum(score_packets, to_nodes, num_segments=N)
    new_score = new_score * d + (1 - d) / N
    return new_score
_jax_for_body_simple = jax.jit(_jax_for_body_simple, static_argnums=(0,))


def pagerank_sparse_jax(N, num_iterations=100, d=0.85):
    from_nodes, to_nodes = flat_adjacency_list(N)
    neighbor_counts = get_neighbor_count(N)

    score = jax.numpy.ones([N], dtype=np.float32) / N
    for _ in range(num_iterations):
        score = _jax_for_body_simple(N, d, score, from_nodes, to_nodes, neighbor_counts)
    # Must cast to np.array to work around JAX's lazy return semantics.
    return np.array(score)


def _jax_for_loop(num_iterations, N, d, from_nodes, to_nodes, neighbor_counts):
    score = jax.numpy.ones([N], dtype=jax.numpy.float32) / N
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
    neighbor_counts = get_neighbor_count(N)

    score = _jax_for_loop(num_iterations, N, d, from_nodes, to_nodes, neighbor_counts)[0]
    # Must cast to np.array to work around JAX's lazy return semantics.
    return np.array(score)


# experimental_compile enables XLA for TF, putting it on even footing with JAX.
@tf.function(experimental_compile=True)
def _tf_for_loop(num_iterations, N, d, from_nodes, to_nodes, neighbor_counts):
    # TF won't let me gather using int16 indices?...
    from_nodes = tf.cast(from_nodes, tf.int32)
    to_nodes = tf.cast(to_nodes, tf.int32)
    # TF can't do float32 divisions by int16s?
    neighbor_counts = tf.cast(neighbor_counts, tf.float32)
    score = tf.ones([N], dtype=tf.float32) / N
    for _ in tf.range(num_iterations):
        score /= neighbor_counts
        score_packets = tf.gather(score,from_nodes)
        score = tf.math.unsorted_segment_sum(score_packets, to_nodes, N)
        score = score * d + (1 - d) / N
    return score

def pagerank_sparse_tf(N, num_iterations=100, d=0.85):
    from_nodes, to_nodes = flat_adjacency_list(N)
    neighbor_counts = get_neighbor_count(N)

    score = _tf_for_loop(num_iterations, N, d, from_nodes, to_nodes, neighbor_counts)
    return np.array(score)


time_result = collections.namedtuple(
    'time_result', ['nodes', 'algorithm', 'time'])

def time_run(graph_size, algorithm, num_trials=10):
    # Prime the lru_caches and JIT caches so that all algorithms are on even footing
    algorithm(graph_size)
    time = timeit.timeit(lambda: algorithm(graph_size), number=num_trials) / num_trials
    return time_result(graph_size, algorithm.__name__, time)


def run_all(algorithms, graph_sizes):
    results = []
    for algo in algorithms:
        print("Benchmarking {}".format(algo))
        for graph_size in graph_sizes:
            num_trials = min(100, max(3, 50000 // graph_size))
            result = time_run(graph_size, algo, num_trials=num_trials)
            results.append(result)
    print(',\n'.join(map(repr, results)))
    # I'm using Altair to plot results, but rendering those plots requires
    # installing selenium or node.js, so we'll rely on Colab to render...
    # Copy this code snippet into Colab and execute to generate plots.
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
    pd.DataFrame({'nodes': endpoints, 'time': 0.00001 * endpoints, 'scaling': 'linear'}),
    pd.DataFrame({'nodes': endpoints3, 'time': 0.0000000001 * endpoints3 ** 3, 'scaling': 'cubic'}),
])
def chart_algorithms(algos):
  subdf = data[data.algorithm.isin(algos)]
  dashed_lines = alt.Chart(overlay_df).mark_line().encode(
      x = alt.X('nodes', scale=alt.Scale(type='log', domain=(10, 30000))),
      y=alt.Y('time', scale=alt.Scale(type='log', domain=(1e-5, 1e2))),
      strokeDash=alt.StrokeDash('scaling', scale=alt.Scale(domain=['linear', 'cubic'], range=[[2, 2], [1, 4]])),
      color=alt.value('black'),
  )

  data_chart = alt.Chart(subdf).mark_line().encode(
      x=alt.X('nodes', scale=alt.Scale(type='log', domain=(10, 30000))),
      y=alt.Y('time', scale=alt.Scale(type='log', domain=(1e-5, 1e2))),
      color=alt.Color('algorithm', sort=[
          'pagerank_dense_np',
          'pagerank_naive',
          'pagerank_sparse_np',
          'pagerank_sparse_jax',
          'pagerank_sparse_np_bincount',
          'pagerank_sparse_jax_rolled',
          'pagerank_sparse_tf',
          'pagerank_sparse_numba',
          'pagerank_sparse_c',
      ])
  )
  return dashed_lines + data_chart
"""

"""
chart_algorithms([
    'pagerank_dense_np',
    'pagerank_naive',
    'pagerank_sparse_np_bincount',
    'pagerank_sparse_c',
    'pagerank_sparse_tf',
    'pagerank_sparse_jax_rolled',
    'pagerank_sparse_numba',
])
"""

"""
normalized_to_c = data.copy()
c_times = normalized_to_c[normalized_to_c['algorithm'] == 'pagerank_sparse_c']
for num_nodes, c_time in c_times[['nodes', 'time']].itertuples(index=False):
  normalized_to_c.loc[normalized_to_c.nodes == num_nodes, 'time'] /= c_time
normalized_to_c.rename(columns={'time': 'times_slower_than_C'}, inplace=True)
alt.Chart(normalized_to_c).mark_line().encode(
    x=alt.X('nodes', scale=alt.Scale(type='log')),
    y=alt.Y('times_slower_than_C', scale=alt.Scale(type='log')),
    color=alt.Color('algorithm', sort=[
        'pagerank_dense_np',
        'pagerank_sparse_py',
        'pagerank_sparse_jax',
        'pagerank_sparse_np_bincount',
        'pagerank_sparse_jax_rolled',
        'pagerank_sparse_tf',
        'pagerank_sparse_c',
    ])
)
"""

if __name__ == '__main__':
    graph_sizes = (10, 30, 100, 300, 1000, 3000, 10000, 30000)
    algorithms = (
        pagerank_sparse_numba,
        pagerank_dense_np,
        pagerank_naive,
#        pagerank_sparse_np,
        pagerank_sparse_np_bincount,
#        pagerank_sparse_jax,
        pagerank_sparse_jax_rolled,
        pagerank_sparse_tf,
    )
    run_all(algorithms, graph_sizes)
