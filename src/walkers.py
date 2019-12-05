"""Walker classes."""

import random
import numpy as np
import networkx as nx
from tqdm import tqdm

class FirstOrderRandomWalker:
    """
    Class to do fast first-order random walks.
    """
    def __init__(self, graph, args):
        """
        Constructor for FirstOrderRandomWalker.
        :param graph: Nx graph object.
        :param args: Arguments object.
        """
        self.graph = graph
        self.walk_length = args.walk_length
        self.walk_number = args.walk_number
        self.walks = []

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.
        :param node: Source node of the truncated random walk.
        :return walk: A single random walk.
        """
        walk = [node]
        for _ in range(self.walk_length-1):
            nebs = [node for node in self.graph.neighbors(walk[-1])]
            if len(nebs) > 0:
                walk = walk + random.sample(nebs, 1)
        walk = [str(w) for w in walk]
        return walk

    def do_walks(self):
        """
        Doing a fixed number of truncated random walk from every node in the graph.
        """
        print("\nModel initialized.\nRandom walks started.")
        for iteration in range(self.walk_number):
            print("\nRandom walk round: "+str(iteration+1)+"/"+str(self.walk_number)+".\n")
            for node in tqdm(self.graph.nodes()):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
        return self.walks

class SecondOrderRandomWalker:
    """
    Class to do second-order random walks.
    """
    def __init__(self, nx_G, is_directed, args):
        """
        Constructor for SecondOrderRandomWalker.
        :param  nx_G: Nx graph object.
        :param is_directed: Directed nature of the graph -- True/False.
        :param args: Arguments object.
        """
        self.G = nx_G
        self.nodes = nx.nodes(self.G)
        print("Edge weighting.\n")
        for edge in tqdm(self.G.edges()):
            self.G[edge[0]][edge[1]]['weight'] = 1.0
            self.G[edge[1]][edge[0]]['weight'] = 1.0
        self.is_directed = is_directed
        self.walk_length = args.walk_length
        self.walk_number = args.walk_number
        self.p = args.P
        self.q = args.Q

    def node2vec_walk(self, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_n = sorted(G.neighbors(cur))
            if len(cur_n) > 0:
                if len(walk) == 1:
                    walk.append(cur_n[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    pre = walk[-2]
                    next = cur_n[alias_draw(alias_edges[(pre, cur)][0], alias_edges[(pre, cur)][1])]
                    walk.append(next)
            else:
                break
        walk = [str(w) for w in walk]
        return walk

    def do_walks(self):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_it in range(self.walk_number):
            print("\nRandom walk round: "+str(walk_it+1)+"/"+str(self.walk_number)+".\n")
            random.shuffle(nodes)
            for node in tqdm(nodes):
                walks.append(self.node2vec_walk(start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        print("")
        print("Preprocesing.\n")
        for node in tqdm(G.nodes()):
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in tqdm(G.edges()):
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []

    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
