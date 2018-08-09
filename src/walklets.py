import networkx as nx
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from joblib import Parallel, delayed
from helper import walk_transformer, create_graph

class WalkletMachine:
    """
    Walklet multi-scale graph factorization machine class.
    The graph is being parsed up, random walks are initiated, embeddings are fitted, concatenated and the multi-scale embedding is dumped to disk.
    """
    def __init__(self, args):
        """
        Walklet machine constructor.
        :param args: Arguments object with the model hyperparameters. 
        """
        self.args = args
        self.graph = create_graph(self.args.input)
        self.walks = []
        self.do_walks()
        self.create_embedding()
        self.save_model()

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.
        :param node: Source node of the truncated random walk.
        :return walk: A single random walk.
        """
        walk = [node]
        for step in range(self.args.walk_length-1):
            nebs = [node for node in self.graph.neighbors(walk[-1])]
            if len(nebs)>0:
                walk = walk + random.sample(nebs,1) 
        walk = map(lambda x: str(x), walk)
        return walk

    def do_walks(self):
        """
        Doing a fixed number of truncated random walk from every node in the graph.
        """
        print("\nModel initialized.\nRandom walks started.")
        for iteration in range(self.args.walk_number):
            print("\nRandom walk round: "+str(iteration+1)+"/"+str(self.args.walk_number)+".\n")
            for node in tqdm(self.graph.nodes()):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)

    def walk_extracts(self, length):
        """
        Extracted walks with skip equal to the length.
        :param length: Length of the skip to be used.
        :return good_walks: The attenuated random walks.
        """
        good_walks = [walk_transformer(walk, length) for walk in self.walks]
        good_walks = [w for walks in good_walks for w in walks]
        return good_walks

    def get_embedding(self, model):
        """
        Extracting the embedding according to node order from the embedding model.
        :param model: A Word2Vec model after model fitting.
        :return embedding: A numpy array with the embedding sorted by node IDs.
        """
        embedding = []
        for node in range(0,len(self.graph.nodes())):
            embedding.append(list(model[str(node)]))
        embedding = np.array(embedding)
        return embedding



    def create_embedding(self):
        """
        Creating a multi-scale embedding.
        """
        self.embedding = []

        for index in range(1,self.args.window_size+1):
            print("\nOptimization round: " +str(index)+"/"+str(self.args.window_size)+".")
            print("Creating documents.")
            clean_documents = self.walk_extracts(index)
            print("Fitting model.")
            model = Word2Vec(clean_documents,
                            size = self.args.dimensions,
                            window = 1,
                            min_count = self.args.min_count,
                            sg = 1,
                            workers = self.args.workers)

            new_embedding = self.get_embedding(model)
            self.embedding = self.embedding +[new_embedding]
        self.embedding = np.concatenate(self.embedding, axis = 1)

    def save_model(self):
        """
        Saving the embedding as a csv with sorted IDs.
        """
        print("\nModels are integrated to be multi scale.\nSaving to disk.")
        self.column_names = map(lambda x: "x_" + str(x), range(self.embedding.shape[1]))
        self.embedding = pd.DataFrame(self.embedding, columns = self.column_names)
        self.embedding.to_csv(self.args.output, index = None)     

