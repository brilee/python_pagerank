import collections
import copy
import functools
import resource
import sys
import timeit
from typing import List, Tuple

import numpy as np
import networkx as nx


@functools.lru_cache()
def make_graph(num_nodes, seed=0):
    g = nx.scale_free_graph(num_nodes, seed=seed)
    for i in range(num_nodes):
        g.add_edge(i, i)
    return g


@functools.lru_cache()
def adjacency_list(g) -> List[List[int]]:
    adjacencies = [[] for _ in range(g.number_of_nodes())]
    for from_edge, to_edge, unused_edge_data in nx.convert.to_edgelist(g):
        adjacencies[from_edge].append(to_edge)
    return adjacencies


@functools.lru_cache()
def adjacency_matrix(g) -> np.ndarray:
    return nx.adjacency_matrix(g).toarray()


@functools.lru_cache()
def flat_adjacency_list(g) -> Tuple[np.ndarray, np.ndarray]:
    from_nodes, to_nodes, _ = zip(*nx.convert.to_edgelist(g))
    return np.array(from_nodes, dtype=np.int16), np.array(to_nodes, dtype=np.int16)


def pagerank_naive(g, num_iterations=100, d=0.85):
    N = g.number_of_nodes()
    for node, node_data in g.nodes(data=True):
        node_data['score'] = 1.0 / N
        node_data['new_score'] = 0
    
    adj_list = adjacency_list(g)

    for _ in range(num_iterations):
        for node, out_nodes in enumerate(adj_list):
            score_packet = g.nodes[node]['score'] / len(out_nodes)
            for out_node in out_nodes:
                g.nodes[out_node]['new_score'] += score_packet
        for node, node_data in g.nodes(data=True):
            node_data['score'] = node_data['new_score'] * d + (1 - d) / N
            node_data['new_score'] = 0
    return np.array([node_data['score'] for node, node_data in g.nodes(data=True)])


def pagerank_dense(g, num_iterations=100, d=0.85):
    adj_matrix = adjacency_matrix(g)
    N = g.number_of_nodes()
    transition_matrix = adj_matrix / np.sum(adj_matrix, axis=1, keepdims=True)
    transition_matrix = (d * transition_matrix + (1 - d) / N).astype(np.float32)

    score = np.ones([N], dtype=np.float32) / N
    for _ in range(num_iterations):
        score = score @ transition_matrix
    return score

def pagerank_sparse(g, num_iterations=100, d=0.85):
    from_nodes, to_nodes = flat_adjacency_list(g)
    neighbor_counts = np.array([len(l) for l in adjacency_list(g)], dtype=np.int16)
    N = g.number_of_nodes()

    score = np.ones([N], dtype=np.float32) / N
    for _ in range(num_iterations):
        score /= neighbor_counts
        score_packets = score[from_nodes]
        score = np.zeros([N], dtype=np.float32)
        np.add.at(score, to_nodes, score_packets)
        score *= d
        score += (1 - d) / N
    return score


time_result = collections.namedtuple(
    'time_result', ['nodes', 'edges', 'algorithm', 'time'])

def time_run(graph_size, algorithm, num_trials=10):
    g = make_graph(graph_size)
    # Prime the lru_caches so that all algorithms are on even footing
    adjacency_list(g)
    adjacency_matrix(g)
    flat_adjacency_list(g)
    time = timeit.timeit(lambda: algorithm(g), number=num_trials) / num_trials
    nodes, edges = g.number_of_nodes(), g.number_of_edges()
    return time_result(nodes, edges, algorithm.__name__, time)


def dump_graphs(graph_sizes):
    """Generate text files with graph data, for use by pagerank.c"""
    for graph_size in graph_sizes:
        print('Generating graph of size %s' % graph_size)
        g = make_graph(graph_size)
        with open('%s.txt' % graph_size, 'w') as f:
            f.write('%s %s\n' % (g.number_of_nodes(), g.number_of_edges()))
            for node, neighbor_list in enumerate(adjacency_list(g)):
                for neighbor in neighbor_list:
                    f.write('%s %s\n' % (node, neighbor))


def run_all(graph_sizes):
    algorithms = (pagerank_naive, pagerank_sparse, pagerank_dense)
    results = []
    for graph_size in graph_sizes:
        num_trials = min(100, max(3, 50000 // graph_size))
        for algo in algorithms:
            result = time_run(graph_size, algo, num_trials=num_trials)
            print(result)
            results.append(result)
    print()
    print(results)
    # Copy data into Colab;  execute the following to generate plots.
    # Altair requires installing selenium or node.js to generate SVGs, so just use Colab...
"""
import collections
import altair as alt
import numpy as np
import pandas as pd
time_result = collections.namedtuple(
    'time_result', ['nodes', 'edges', 'algorithm', 'time'])
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
    if len(sys.argv) == 1: 
        print(pagerank_dense(make_graph(10)))
#        run_all(graph_sizes)
    elif sys.argv[1] == 'dump_graphs':
        dump_graphs(graph_sizes)
