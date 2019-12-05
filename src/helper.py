"""Utils for data reading and writing."""

import argparse
import pandas as pd
import networkx as nx

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Facebook food dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """
    parser = argparse.ArgumentParser(description="Run Walklet.")

    parser.add_argument("--input",
                        nargs="?",
                        default="./input/food_edges.csv",
	                help="Input folder with jsons.")

    parser.add_argument("--output",
                        nargs="?",
                        default="./output/food_embedding.csv",
	                help="Embeddings path.")

    parser.add_argument("--walk-type",
                        nargs="?",
                        default="second",
	                help="Random walk order.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=16,
	                help="Number of dimensions. Default is 16.")

    parser.add_argument("--walk-number",
                        type=int,
                        default=5,
	                help="Number of walks. Default is 5.")

    parser.add_argument("--walk-length",
                        type=int,
                        default=80,
	                help="Walk length. Default is 80.")

    parser.add_argument("--window-size",
                        type=int,
                        default=5,
	                help="Number of embeddings. Default is 5.")

    parser.add_argument("--workers",
                        type=int,
                        default=4,
	                help="Number of cores. Default is 4.")

    parser.add_argument("--min-count",
                        type=int,
                        default=1,
	                help="Minimal appearance feature count. Default is 1.")

    parser.add_argument("--P",
                        type=float,
                        default=1.0,
	                help="Return hyperparameter. Default is 1.0.")

    parser.add_argument("--Q",
                        type=float,
                        default=1.0,
	                help="Inout hyperparameter. Default is 1.0.")

    return parser.parse_args()

def create_graph(file_name):
    """
    Reading an edge list csv and returning an Nx graph object.
    :param file_name: location of the csv file.
    :return graph: Networkx graph object.
    """
    edges = pd.read_csv(file_name).values.tolist()
    graph = nx.from_edgelist(edges)
    return graph

def walk_transformer(walk, length):
    """
    Tranforming a given random walk to have skips.
    :param walk: Random walk as a list.
    :param length: Skip size.
    :return transformed_walk: Walk chunks for training.
    """
    transformed_walk = []
    for step in range(length+1):
        neighbors = [y for i, y in enumerate(walk[step:]) if i % length == 0]
        transformed_walk.append(neighbors)
    return transformed_walk
