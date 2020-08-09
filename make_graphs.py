import numpy as np
import networkx as nx


def make_graph(num_nodes, seed=0):
    g = nx.scale_free_graph(num_nodes, seed=seed)
    for i in range(num_nodes):
        g.add_edge(i, i)
    return g


def dump_graphs(graph_sizes):
    """Cache generated graph data, for use by pagerank.c."""
    for graph_size in graph_sizes:
        print('Generating graph of size %s' % graph_size)
        g = make_graph(graph_size)
        with open('%s.txt' % graph_size, 'w') as f:
            f.write('%s %s\n' % (g.number_of_nodes(), g.number_of_edges()))
            for from_node, to_node, _ in nx.convert.to_edgelist(g):
                    f.write('%s %s\n' % (from_node, to_node))



if __name__ == '__main__':
    graph_sizes = (10, 30, 100, 300, 1000, 3000, 10000, 30000)
    dump_graphs(graph_sizes)
