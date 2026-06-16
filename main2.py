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
    print(healing_factor)
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
    
    
def sdrg_step(G, mu_arr, lambda_arr):
    n_nodes = G.numberOfNodes()

    max_mu_lambda_index = np.argmax(np.concat((mu_arr, lambda_arr[0,:])))
    #print(max_mu_lambda_index)
    if max_mu_lambda_index > n_nodes: #true iff biggest value is a transmittal rate
        edge_to_decimate = lambda_arr[1:, max_mu_lambda_index-n_nodes] #np list of length 2
        
        #union the two sets containing neighbors of u and v
        # sets automatically remove the duplicates
        pair_neighborhood = {u for u in G.iterNeighbors(edge_to_decimate[0])} | {v for v in G.iterNeighbors(edge_to_decimate[1])}
        #neighbors_checked = set(())
        u = edge_to_decimate[0]
        v = edge_to_decimate[1]
        

        k = G.addNode() # returns new node id, so k = new node id
        healing_factor = G.getNodeAttribute("mu", float)
        
        #use log and exp to convert mults and divides to adds and substracts
        h_k = math.exp(math.log(healing_factor[u]) + math.log(healing_factor[v]) - math.log(G.weight(u,v)))
        np.append(mu_arr, h_k)
        print(healing_factor)
        
        healing_factor[k] = h_k
        
        for neighbor in pair_neighborhood:
            #we are merging nodes u and v
            # i for each neighbor
            J_ui = G.weight(u, neighbor)
            J_vi = G.weight(v, neighbor)
            
            new_edge_weight = max(J_ui, J_vi)
            G.addEdge(k, neighbor, new_edge_weight)
            np.append(lambda_arr, np.array([new_edge_weight, k, neighbor]).T)
            
        #remove nodes at end. We had to wait b/c otherwise we can't calculate weights in loop
        G.removeNode(u)
        G.removeNode(v)
        mu_arr = mu_arr[mu_arr != u and mu_arr != v]
        
        print(f"decimated edge between nodes {u} and {v}")
        
        
        #print(edge_to_decimate)
        
        
    else: #true iff biggest val is a mu
        site_to_decimate = max_mu_lambda_index
        neighborhood = {u for u in G.iterNeighbors(site_to_decimate)}
        neighbors_checked = set(())
        for neighbor in neighborhood:
            if (neighborhood - neighbors_checked).remove(neighbor):
                for other_neighbor in (neighborhood - neighbors_checked).remove(neighbor):
                    # before change to sets it was this: np.delete(neighborhood, neighbor).delete(neighbors_checked): 
                    # i = decimated site 
                    # j, k = neighbors
    
                    J_jk = G.weight(neighbor, other_neighbor)
                    r_hi = -math.log(mu_arr[site_to_decimate])
                    kappa_ik = -math.log(G.weight(site_to_decimate, other_neighbor))
                    kappa_ij = -math.log(G.weight(neighbor, site_to_decimate))
                    
                    new_weight = max(J_jk, math.exp(kappa_ik + kappa_ij - r_hi))
                    if J_jk > 0:
                        
                    G.setWeight(neighbor, other_neighbor, new_weight)
                    
            #The neighbor that just looped through all of the other neighbors will
            # have had all of its connections made and calculated, so no reason to 
            # do anything to it for the rest of the loop
            
            neighbors_checked.add(neighbor)
            #np.append(neighbors_checked, neighbor)
        G.removeNode(site_to_decimate)
        
        print(f"decimated site {site_to_decimate}")
    
    return G, mu_arr, lambda_arr
            
            
    #lambda_max = np.max(lambda_arr)
    
    
    
    
    
    
#x = sdrg(*generate_graph(500, 50, 20, 0.1, 0.01))

