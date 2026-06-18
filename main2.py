# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:00:04 2026

@author: zidda
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
#import matplotlib as plt
import networkit as nk
from networkit import vizbridges
import time
import matplotlib.pyplot as plt

np.random.seed(1)


#G = nx.partial_duplication_graph(50, 4, 0.4, 0.6)

def generate_square_lattice(width, height, torus = True, visualize_on=False, disordered_mu = False, constant_mu = 1.0, disordered_lambda=False, constant_lambda = 0.6):
    G = nk.graph.Graph(n=width*height, weighted = True, edgesIndexed=True)
    
    if not torus:
        #first build a bunch of lines/rows
        for y in range(height):
            for x in range(1, width):  
                this_node = x+(y*width)
                G.addEdge(this_node, (this_node - 1))
        #connect the lines to form a grid
        for y in range(1, height):
            for x in range(width):  
                this_node = x+(y*width)
                G.addEdge(this_node, (this_node - width))
    
    if torus:
        #first build a bunch of lines/rows
        for y in range(height):
            for x in range(1, width):  
                this_node = x+(y*width)
                G.addEdge(this_node, (this_node - 1))
        #connect the lines to form a grid
        for y in range(1, height):
            for x in range(width):  
                this_node = x+(y*width)
                G.addEdge(this_node, (this_node - width))
        
        #connect sides
        for x in range(width):  
            this_node = x
            G.addEdge(this_node, (this_node + (width*(height-1))))
        
        #connect top/bottom
        for y in range(height):
            this_node = y*width
            G.addEdge(this_node, (this_node+width-1))
                
    
    if visualize_on:
        print("generated; now visualizing")
        visualize(G)
    
    #G.indexEdges()
    
    if disordered_lambda:
        nk.graphtools.randomizeWeights(G) #these weights act as lambda vals
    else:
        for u,v in G.iterEdges():
            G.setWeight(u,v,constant_lambda)
    
    #init activity values (infected)
    is_active = G.attachNodeAttribute("active", int)
    
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)

    #add disorder to healing values too if desired
    if disordered_mu:
        for u in G.iterNodes():
            mu = np.random.random()
            healing_factor[u] = mu
            is_active[u] = 0
            
    else:
        for u in G.iterNodes():
            healing_factor[u] = constant_mu
            is_active[u] = 0
            
    print("done!")
        
    return G



def generate_arbitrary_graph(size, num_clusters, p_in, p_out):
    #Gen random clustered graph
    G = nk.generators.ClusteredRandomGraphGenerator(size, num_clusters, p_in, p_out).generate()
    G.indexEdges()
    nk.graphtools.randomizeWeights(G) #these weights act as lambda vals
    
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    print(healing_factor)
    #transmittal_rate = G.attachEdgeAttribute("lamda", float)
    for u in G.iterNodes():
        mu = np.random.random()
        healing_factor[u] = mu
    #
    """
    mu_dict = {u: np.random.random() for u in G.iterNodes()}#np.zeros(G.numberOfNodes())
    lambda_dict = {(u,v): np.random.random() for u,v in G.iterEdges()}#np.zeros((3, G.numberOfEdges()))
    for u in G.iterNodes():
        mu = np.random.random()
        healing_factor[u] = mu
        #mu_arr[u] = mu
    
    edge_iter = 0
    #print(lambda_arr[:,5])
    for u,v in G.iterEdges():
        lambda_arr[:, edge_iter] = np.array([G.weight(u,v), u, v]).T
        edge_iter += 1
    #node_atts = {x: (False, np.random.random()) for x in range(len(G.nodes))}
    #print(list(G.edges))
    """
    #nk.graphio.writeGraph(G, "network.gml", nk.Format.GML)
    return G #, mu_arr, lambda_arr
    
    
def sdrg_step(G, visualizeStep = False):
    n_nodes = G.numberOfNodes()
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    if n_nodes == 2:
        print("Network is fully sparsified")
    
    mu_arr = np.array([np.array([healing_factor[u], u]).T for u in G.iterNodes()])
    lambda_arr = np.array([np.array([G.weight(u,v), u, v]).T for u,v in G.iterEdges()])
    
    
    #print(lambda_arr[:,0])
        
    max_mu_lambda_index = np.argmax(np.concat((mu_arr[:,0], lambda_arr[:,0])))
    #print(max_mu_lambda_index)
    if max_mu_lambda_index > n_nodes-1: #true iff biggest value is a transmittal rate
        edge_to_decimate = lambda_arr[max_mu_lambda_index-n_nodes, 1:] #np list of length 2
        
        #union the two sets containing neighbors of u and v
        # sets automatically remove the duplicates
        pair_neighborhood = {u for u in G.iterNeighbors(edge_to_decimate[0])} | {v for v in G.iterNeighbors(edge_to_decimate[1])}
        #neighbors_checked = set(())
        u = edge_to_decimate[0]
        v = edge_to_decimate[1]
        print(f"decimating edge {edge_to_decimate} with lambda = {G.weight(u,v)}")

        k = G.addNode() # returns new node id, so k = new node id
        
        
        #use log and exp to convert mults and divides to adds and substracts
        h_k = math.exp(math.log(healing_factor[u]) + math.log(healing_factor[v]) - math.log(G.weight(u,v)))
        #np.append(mu_arr, h_k)
        
        healing_factor[k] = h_k
        is_active[k] = 0
        
        for neighbor in pair_neighborhood:
            #we are merging nodes u and v
            # i for each neighbor
            J_ui = G.weight(u, neighbor)
            J_vi = G.weight(v, neighbor)
            
            new_edge_weight = max(J_ui, J_vi)
            G.addEdge(k, neighbor, new_edge_weight)
            #np.append(lambda_arr, np.array([new_edge_weight, k, neighbor]).T)
            
        #remove nodes at end. We had to wait b/c otherwise we can't calculate weights in loop
        G.removeNode(u)
        G.removeNode(v)
        #mu_arr = mu_arr[mu_arr != u and mu_arr != v]
        
        print(f"decimated edge between nodes {u} and {v}")
        
        
        #print(edge_to_decimate)
        
        
    else: #true iff biggest val is a mu
        
        site_to_decimate = mu_arr[max_mu_lambda_index,1]
        print(f"decimating site {site_to_decimate} with mu = {healing_factor[site_to_decimate]}")
        neighborhood = {u for u in G.iterNeighbors(site_to_decimate)}
        neighbors_checked = {-1}
        neighbors_checked.remove(-1)
        for neighbor in neighborhood:
            
            s1 = neighborhood - neighbors_checked
            s1.remove(neighbor)
            if s1:
                for other_neighbor in s1:
                    # before change to sets it was this: np.delete(neighborhood, neighbor).delete(neighbors_checked): 
                    # i = decimated site 
                    # j, k = neighbors
    
                    J_jk = G.weight(neighbor, other_neighbor)
                    r_hi = -math.log(healing_factor[site_to_decimate])
                    kappa_ik = -math.log(G.weight(site_to_decimate, other_neighbor))
                    kappa_ij = -math.log(G.weight(neighbor, site_to_decimate))
                    
                    new_weight = max(J_jk, math.exp(kappa_ik + kappa_ij - r_hi))
                    
                    #G.removeEdge(neighbor, other_neighbor)
                    add_success = G.addEdge(neighbor, other_neighbor, new_weight, checkMultiEdge = True)
                    if not add_success:
                        G.setWeight(neighbor, other_neighbor, new_weight)
                    #G.setWeight(neighbor, other_neighbor, new_weight)
                    
            #The neighbor that just looped through all of the other neighbors will
            # have had all of its connections made and calculated, so no reason to 
            # do anything to it for the rest of the loop
            
            neighbors_checked.add(neighbor)
            #np.append(neighbors_checked, neighbor)
        G.removeNode(site_to_decimate)
        
        
        
        print(f"decimated site {site_to_decimate}")
    #nk.graphio.writeGraph(G, f"network_{time.time_ns()}_T.gml", nk.Format.GML)
    if visualizeStep:
        visualize(G)
    return G

def visualize(G, active_nodes=[],inactive_nodes=[]):
    #G = nk.readGraph("../input/karate.graph", nk.Format.METIS)
    # Initalize and run Betweenness algorithm
    
    nx_graph = nk.nxadapter.nk2nx(G)

    # 3. Draw the graph using Matplotlib
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(nx_graph, seed=2)
    
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw(nx_graph, pos=pos, with_labels=True, node_color='skyblue', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=active_nodes, node_color="tab:red", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=inactive_nodes, node_color="tab:blue", **options)
    plt.show()
    
    
def DCP(G, t_max = 1, random_start=True, lattice = True, torus = True):
    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    nodes = list(G.iterNodes())
    if random_start:
        s = np.random.randint(0, len(nodes))
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
    
    N_active = 1
    
    step_counter = 0 #used just to draw updates every once in a while
    
    """
    We implement the SIS algorithm described below from 
    (https://journals.aps.org/pre/abstract/10.1103/PhysRevE.86.041125) 
    Epidemic thresholds of the susceptible-infected-susceptible model on 
    networks: A comparison of numerical and theoretical results
    ---------------------------------------------------------------------
    We consider here the SIS model for epidemics in continuous time. The 
    numerical algorithm is implemented as follows: At each time step, we 
    compute the number of infected nodes, Ni, and links emanating from them, 
    Nn. With probability Ni/(Ni+λNn), one infected node, chosen at random, 
    becomes healthy. With complementary probability λNn/(Ni+λNn), one of the 
    links is selected uniformly at random and the infection is transmitted 
    through it from the infected node corresponding to one of the ends of the 
    edge, toward the (possibly susceptible) node at the other end. The numbers 
    of infected nodes and related links are updated accordingly, time is 
    incremented by Δt=1/(Ni+λNn), and the whole process is iterated.
    """
    t = 0
    
    while t < t_max:
        active_nodes = np.array([u for u in G.iterNodes() if is_active[u] == 1])
        exposed_edges = np.array([
                np.array([u, v]).T 
                for u in active_nodes #outer
                for v in G.iterNeighbors(u) #inner
            ])
            
        N_active = len(active_nodes)
        N_exposed = len(exposed_edges)# the number of exposed nodes is the same as the number of edges connected to active sites
        lambda_avg = (G.totalEdgeWeight()/G.numberOfEdges())
        #print(active_nodes)
        u1 = np.random.random()
        prob_to_heal = N_active/(N_active+(lambda_avg*N_exposed))
        #print(prob_to_heal)
        if u1 <= prob_to_heal:
            
            site_to_heal = np.random.randint(0, N_active)
            is_active[active_nodes[site_to_heal]] = 0
            #print(f"healed site {active_nodes[site_to_heal]}")
        else:
            #print(exposed_edges)
            #print(u1)
            #print(patient_zero)
            #edges_to_sum = [[G.weight(u,v) for v in G.iterNeighbors(u)] for u in active_nodes]
            #print(edges_to_sum)
            #print(f"exposededges[0] = {exposed_edges[:]}")
            infection_chances_list = np.array([sum([G.weight(u,v) for v in G.iterNeighbors(u)]) for u in exposed_edges[:,0]])
            infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
            #print(f"active are {active_nodes}")
            exposed_sites = exposed_edges[:, 1]
            #print(exposed_sites)
            #print(infection_chances_list)
    
            site_to_infect = np.random.choice(exposed_sites, p=infection_chances_list)
            is_active[site_to_infect] = 1
            #print(f"infected site {site_to_infect}")
        
        active_nodes = np.array([u for u in G.iterNodes() if is_active[u]])
        print(f"Active: {active_nodes} ---- t = {t}")
            
        t += 1/(N_active+lambda_avg*N_exposed)
        step_counter+=1
        print(step_counter)
        if step_counter%1 == 0:
            visualize(G, active_nodes=active_nodes)
    
    
#x = sdrg(*generate_graph(500, 50, 20, 0.1, 0.01))

