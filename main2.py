# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:00:04 2026

@author: zidda
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as plt
import networkit as nk

np.random.seed(1)
size = 15000
num_clusters = 10
p_in = 0.2
p_out = 0.05
#G = nx.partial_duplication_graph(50, 4, 0.4, 0.6)

def generate_arbitrary_graph(size, num_clusters, p_in, p_out):
    #Gen random clustered graph
    G = nk.generators.ClusteredRandomGraphGenerator(size, num_clusters, p_in, p_out).generate()
    G.indexEdges()
    nk.graphtools.randomizeWeights(G) #these weights act as lambda vals
    
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    #transmittal_rate = G.attachEdgeAttribute("lamda", float)
    
    #
    mu_arr = np.zeros(G.numberOfNodes())
    lambda_arr = np.zeros((3, G.numberOfEdges()))
    for u in G.iterNodes():
        mu = np.random.random()
        healing_factor[u] = mu
        mu_arr[u] = mu
    
    edge_iter = 0
    #print(lambda_arr[:,5])
    for u,v in G.iterEdges():
        lambda_arr[:, edge_iter] = np.array([G.weight(u,v), u, v]).T
        edge_iter += 1
    #node_atts = {x: (False, np.random.random()) for x in range(len(G.nodes))}
    #print(list(G.edges))

    
    

    
    
    return G, mu_arr, lambda_arr
    
    
def sdrg(G, mu_arr, lambda_arr):
    n_nodes = G.numberOfNodes()
    #healing_factor.to_numpy()
    max_mu_lambda_index = np.argmax(np.concat((mu_arr, lambda_arr[0,:])))
    print(max_mu_lambda_index)
    if max_mu_lambda_index > n_nodes: #true iff biggest value is a transmittal rate
        edge_to_decimate = lambda_arr[1:, max_mu_lambda_index-n_nodes]
        print(edge_to_decimate)
    #lambda_max = np.max(lambda_arr)
    
    
    
    
    
    
#x = sdrg(*generate_graph(500, 50, 20, 0.1, 0.01))

