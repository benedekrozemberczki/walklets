import networkx as nx
import glob
import random
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from joblib import Parallel, delayed
from tqdm import tqdm
from helper import walk_transformer, create_graph

class WalkletMachine:
    """
    """
    def __init__(self, args):
        """
        """
        self.args = args
        self.g = create_graph(self.args.input)
        self.walks = []
        self.do_walks()
        self.create_embedding()
        self.save_model()

    def do_walk(self, node):
        """
        """
        walk = [node]
        for step in range(self.args.walk_length-1):
            nebs = [node for node in self.g.neighbors(walk[-1])]
            if len(nebs)>0:
                walk = walk + random.sample(nebs,1) 
        walk = map(lambda x: str(x), walk)
        return walk

    def do_walks(self):
        """
        """
        print("\nModel initialized.\nRandom walks started.")
        for iteration in range(self.args.walk_number):
            print("\nRandom walk round: "+str(iteration+1)+"/"+str(self.args.walk_number)+".\n")
            for node in tqdm(self.g.nodes()):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)

    def walk_extracts(self, length):
        """
        """
        good_walks = [walk_transformer(walk, length) for walk in self.walks]
        good_walks = [w for walks in good_walks for w in walks]
        return good_walks

    def get_embedding(self,model):
        """
        """
        out = []
        for node in range(0,len(self.g.nodes())):
            out.append(list(model[str(node)]))
        return np.array(out)



    def create_embedding(self):
        """
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
        """

        print("\nModels are integrated to be multi scale.\nSaving to disk.")
        self.column_names = map(lambda x: "x_" + str(x), range(self.embedding.shape[1]))
        self.embedding = pd.DataFrame(self.embedding, columns = self.column_names)
        self.embedding.to_csv(self.args.output, index = None)     

