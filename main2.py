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
import time
import matplotlib.pyplot as plt
#import random
from scipy import sparse
from scipy.sparse.linalg import cg
import scipy
from scipy.stats import wasserstein_distance
import bisect


np.random.seed(1)
    #"C:\Users\zidda\Downloads\inputs_L10_pbcTrue_h0-1.0_j0-1.0_seed1.npz"
    
    
""" ------------------------------------ GRAPH GENERATION ------------------------------------ """
    

def translate_Neil_NP_to_graph(filename='inputs_L50_pbcTrue_h0-1.0_j0-1.0_seed1.npz'):
    #This function translates the numpy arrays produced in Neil's program into a networkit graph analyzable by mine
    with np.load(filename) as data:
        print("Keys in file:", data.files)
        L = data[data.files[0]]
        r = data[data.files[1]]
        kappa = data[data.files[2]]
        bonds = data[data.files[3]]
        n_sites = data[data.files[4]]
        n_bonds = data[data.files[5]]
        
        G = nk.graph.Graph(n=n_sites, weighted = True, edgesIndexed = True)
        
        #init activity values (infected/noninfected)
        is_active = G.attachNodeAttribute("active", int)
        
        #init mu values
        healing_factor = G.attachNodeAttribute("mu", float)
        components = G.attachNodeAttribute("components", str)
        
        #for each bond, give it the transmittal rate from Neil's program
        for i in range(len(bonds)):
            G.addEdge(bonds[i,0], bonds[i,1])
            J = math.exp(-1*kappa[i])
            G.setWeight(bonds[i,0], bonds[i,1], J)
            #print(f"J = {J}")
        
        #also do the same for healing rate
        for u in G.iterNodes():
            is_active[u] = False
            h = math.exp(-1*r[u])
            healing_factor[u] = h
            components[u] = f"{u}"
            #print(f"h = {h}")
        
        #visualize(G)
        return G
        
        print(bonds)
        
        
def generate_square_lattice(width, height, torus = True, visualize_on=False, disordered_mu = True, constant_mu = 1.0, disordered_lambda=True, constant_lambda = 0.8):
    #Self explanatory
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
    
    
    if disordered_lambda:
        nk.graphtools.randomizeWeights(G) #these weights act as lambda vals
    else:
        for u,v in G.iterEdges():
            G.setWeight(u,v,constant_lambda)
    
    #init activity values (infected)
    is_active = G.attachNodeAttribute("active", int)
    
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    components = G.attachNodeAttribute("components", str)
    #add disorder to healing values too if desired
    if disordered_mu:
        for u in G.iterNodes():
            mu = np.random.random()
            healing_factor[u] = mu
            is_active[u] = 0
            components[u] = f"{u}"
    else:
        for u in G.iterNodes():
            healing_factor[u] = constant_mu
            is_active[u] = 0
            components[u] = f"{u}"
    
    print("done!")
        
    return G


def generate_chain(length, loop = True, visualize_on=False, disordered_mu = True, constant_mu = 1.0, disordered_lambda=True, constant_lambda = 0.8):
    G = nk.graph.Graph(n=length, weighted = True, edgesIndexed=True)
    
    if not loop:
        for x in range(1, length):
            G.addEdge(x, x - 1)
        
    
    else:
        for x in range(1, length):
            G.addEdge(x, x - 1)
        
        
        G.addEdge(0, length-1)
                
    
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
    components = G.attachNodeAttribute("components", str)
    #add disorder to healing values too if desired
    if disordered_mu:
        for u in G.iterNodes():
            mu = np.random.random()
            healing_factor[u] = mu
            is_active[u] = 0
            components[u] = f"{u}"
    else:
        for u in G.iterNodes():
            healing_factor[u] = constant_mu
            is_active[u] = 0
            components[u] = f"{u}"
    
    print("done!")
        
    return G


def generate_arbitrary_graph(size, num_clusters, p_in, p_out):
    #Gen random clustered graph. This is outdated and won't work for new versions of the code because it doesn't properly assign values to things
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
    

""" ------------------------------------ GRAPH SPARSIFICATION ------------------------------------ """
    

def sdrg_step(G, neil_mode = False, decimated_sites=[], logging_toggle = True, decimation_log = [], visualizeStep = False):
    """This method does a single iteration of the SDRG sparsification. 
        - G: This is a networkit graph. It should be fully-connected/only contain one component, and needs to have edge weights + mu values (healing factor) + components (initialized with each node's own index)
        - neil_mode: Neil has a sparsification algorithm that outputs, at the end, all of the decimated clusters. This toggle just changes to output to this for checking the sdrg results against his
        - decimated_sites: only relevant in neil mode; this stores the sites that are decimated in each step as clusters
        - logging_toggle: This is my own logging; it is at time of writing, not used. Basically just an alternative to neil_mode logging
        - decimation_log: decimation_log is to logging_toggle as decimated_sites is to neil_mode. tracks the edges that get maximum rule-d out each step
        - visualizeStep: visualizes the network AFTER applying the sdrg step using networkx. This gets reeeaaaalllllyyyy slow above like a hundred nodes or so
    """
    
    n_nodes = G.numberOfNodes()
    healing_factor = G.getNodeAttribute("mu", float)
    components = G.getNodeAttribute("components", str)
    is_active = G.getNodeAttribute("active", int)
    
    #Stopping condition
    if n_nodes == 1:
        print("Network is fully sparsified")
        last_site = [u for u in G.iterNodes()]

        decimated_sites = np.append(decimated_sites, components[last_site[0]])
        if logging_toggle:
            return G, decimation_log
        elif neil_mode:
            return G, decimated_sites
        return G
     
    #Put all mu and lambda values into a single np array so that we can easily choose the greatest value
    mu_arr = np.array([np.array([healing_factor[u], u]).T for u in G.iterNodes()])
    lambda_arr = np.array([np.array([G.weight(u,v), u, v]).T for u,v in G.iterEdges()])
            
    max_mu_lambda_index = np.argmax(np.concat((mu_arr[:,0], lambda_arr[:,0])))
    
    
    if max_mu_lambda_index > n_nodes-1: #true iff biggest value is a lambda
        edge_to_decimate = lambda_arr[max_mu_lambda_index-n_nodes, 1:] #np list of length 2
        
        #union the two sets containing neighbors of u and v
        # sets automatically remove the duplicates. This now contains only the nodes which are neighbors of/connected to both
        pair_neighborhood = {u for u in G.iterNeighbors(edge_to_decimate[0])} | {v for v in G.iterNeighbors(edge_to_decimate[1])}

        #give a name to each side of the edge
        u = edge_to_decimate[0]
        v = edge_to_decimate[1]
        #print(f"decimating edge {edge_to_decimate} with lambda = {G.weight(u,v)}") 

        #note that, as a result of this step, this sdrg algorithm creates new node indices up to 2x the original size of the graph. This is important for some methods like fast_random_choose()
        k = G.addNode() # returns new node id, so k = new node id
        #if logging_toggle:
            #decimation_log.append
            #decimation_log.append([len(decimation_log), edge_to_decimate, G.weight(u,v)])
        
        #TODO: use log and exp to convert mults and divides to adds and substracts
        #calculate a new healing factor
        #math.exp(math.log(healing_factor[u]) + math.log(healing_factor[v]) - math.log(G.weight(u,v)))
        h_k = (healing_factor[u]*healing_factor[v])/(G.weight(u,v))

        healing_factor[k] = h_k
        #keep track of our components
        components[k] = f"{components[u]}_{components[v]}"
        is_active[k] = 0
        
        for neighbor in pair_neighborhood:
            #we are merging nodes u and v
            # i for each neighbor
            J_ui = G.weight(u, neighbor)
            J_vi = G.weight(v, neighbor)
            
            new_edge_weight = max(J_ui, J_vi)
            if logging_toggle:
                if J_ui == new_edge_weight:
                    decimation_log.append((v,neighbor))
                else:
                    decimation_log.append((u,neighbor))
            G.addEdge(k, neighbor, new_edge_weight)
            
        #remove nodes at end. We had to wait b/c otherwise we can't calculate weights in loop
        G.removeNode(u)
        G.removeNode(v)
        
    else: #true iff biggest val is a mu
        
        #since we put the mu_arr in front of the lambda_arr, we can directly access the mu_arr list with our max_mu_lambda_index without getting indexing errors
        site_to_decimate = mu_arr[max_mu_lambda_index,1]
        #print(f"decimating site {site_to_decimate} with mu = {healing_factor[site_to_decimate]}")
        #build a set with all of the neighbors
        neighborhood = {u for u in G.iterNeighbors(site_to_decimate)}
        
        #I've forgotten why I did this, it seems kind of stupid but I don't really want to bother changing it right now. 
        #it can probably be replaced by neighbors_checked = set()
        neighbors_checked = {-1}
        neighbors_checked.remove(-1)
        
        for neighbor in neighborhood:
            s = neighborhood - neighbors_checked # s is the set of unchecked neighbors. For high-degree networks this will be a significant speedup but probably not so for lattices
            s.remove(neighbor) #because we've already checked it
            if s: #... has any elements
                for other_neighbor in s:
                    # before change to sets it was this: np.delete(neighborhood, neighbor).delete(neighbors_checked): 
                    # i = decimated site 
                    # j, k = neighbors
    
                    J_jk = G.weight(neighbor, other_neighbor)
                    #r_hi = -math.log(healing_factor[site_to_decimate])
                    #kappa_ik = -math.log(G.weight(site_to_decimate, other_neighbor))
                    #kappa_ij = -math.log(G.weight(neighbor, site_to_decimate))
                    
                    #this is a long expression so I make it a variable. It's really just lambda_ui*lambda_uj/mu_u
                    a = (G.weight(site_to_decimate, other_neighbor)*G.weight(neighbor, site_to_decimate))/healing_factor[site_to_decimate]
                    
                    new_weight = max(J_jk, a)
                    if logging_toggle:
                        if J_jk == new_weight:
                            decimation_log.append((site_to_decimate, other_neighbor))
                            decimation_log.append((site_to_decimate, neighbor))
                        else:
                            decimation_log.append((neighbor, other_neighbor))
                    #returns whether the addition was successful (i.e, didn't make a multiedge)
                    add_success = G.addEdge(neighbor, other_neighbor, new_weight, checkMultiEdge = True)
                    if not add_success:
                        G.setWeight(neighbor, other_neighbor, new_weight)
                    #G.setWeight(neighbor, other_neighbor, new_weight)
                    
            #The neighbor that just looped through all of the other neighbors will
            # have had all of its connections made and calculated, so no reason to 
            # do anything to it for the rest of the loop. This just preserves the 
            # action done in the line with s.remove(neighbor) for future loops
            neighbors_checked.add(neighbor)
            
        decimated_sites = np.append(decimated_sites, components[site_to_decimate])
        #print(f"decimated site {site_to_decimate} with components {components[site_to_decimate]}")
        if logging_toggle:
            #decimation_log.append([len(decimation_log), site_to_decimate, healing_factor[site_to_decimate]])
            pass
        #finally we get to actually remove the node
        G.removeNode(site_to_decimate)
        
    #nk.graphio.writeGraph(G, f"network_{time.time_ns()}_T.gml", nk.Format.GML)
    if visualizeStep:
        visualize(G)
    if logging_toggle:
        return G, decimation_log
    elif neil_mode:
        return G, decimated_sites
    return G

def sdrg_to_completion(G, visualizeSteps = False):
    decimation_log = []
    for i in range(G.numberOfNodes()):
        G, decimation_log = sdrg_step(G, decimation_log=decimation_log, visualizeStep=visualizeSteps)
    #print(decimated_sites)
    print("--------------------------")
        
    return G, decimation_log

def full_sdrg(G, visualizeSteps = False):
    decimated_sites = []
    for i in range(G.numberOfNodes()):
        G, decimated_sites = sdrg_step(G, decimated_sites, visualizeStep=visualizeSteps)
        
    #print(decimated_sites)
    print("--------------------------")
        
    return G


def n_sdrg(G, n, visualizeSteps = False):
    decimated_sites = []
    for i in range(n):
        G, decimated_sites = sdrg_step(G, decimated_sites, visualizeStep=visualizeSteps)
        
    #print(decimated_sites)
    print("--------------------------")
        
    return G


def until_n_remain_sdrg(G, n, visualizeSteps = False):
    #
    decimated_sites = []
    for i in range(G.numberOfNodes()-n+1):
        G, decimated_sites = sdrg_step(G, decimated_sites, visualizeStep=visualizeSteps)
        
    #print(decimated_sites)
    print("--------------------------")
        
    return G


def until_n_components(G, n, visualizeSteps = False):
    decimated_sites=[]
    while number_of_components(G) > n:
        G, decimated_sites = sdrg_step(G, decimated_sites, visualizeStep=visualizeSteps)
        #print(number_of_components(G))
    return G


def prop_sdrg(G, proportion, visualizeSteps = False):
    """
    This function keeps <proportion> percent of the network's nodes
    """
    n = G.numberOfNodes()-round(G.numberOfNodes()*proportion)
    decimated_sites = []
    for i in range(n):
        G, decimated_sites = sdrg_step(G, decimated_sites, visualizeStep=visualizeSteps)
        if i%100 ==0:
            print(f"decimated {i} nodes")
    #print(decimated_sites)
    print("--------------------------")
        
    return G


def sdrg_sparsify(G, proportion):
    G_tilde_1 = copy_graph(G)
    G_tilde_2 = copy_graph(G)
    G_tilde_1, decimation_log = sdrg_to_completion(G_tilde_1)
    for i in range(len(decimation_log)):
        edge = decimation_log[i]
        G_tilde_2.removeEdge(edge[0], edge[1])
    



def sampling_sparsification(G, n_samples, method = 'u', visualize_on = False):
    """
    Methods are uniform, weight-based, and effective resistance; corresponding to arguments of 'u', 'w', and 'er'
    """
    G_tilde = nk.graph.Graph(n=G.numberOfNodes(), weighted=True, edgesIndexed=True)
    edges = np.array([(u,v) for u,v in G.iterEdges()])
    
    if method == 'u':
        edge_pmf = np.ones(len(edges)) * (1/len(edges))
    elif method == 'w':
        pass
        

def uniform_sampling_sparsification(G, n_samples, visualize_on = False):
    G_tilde = nk.graph.Graph(n=G.numberOfNodes(), weighted=True, edgesIndexed=True)
    edges = np.array([(u,v) for u,v in G.iterEdges()])
    
    for sample in range(n_samples):
        choice_index = int(np.random.choice(np.arange(edges.size/2)))
        choice = edges[choice_index]
        
        #we ensure that there are no duplicates
        if G_tilde.weight(choice[0], choice[1]) == 0:
            new_weight = G.weight(choice[0], choice[1]) 
        else:
            new_weight = G_tilde.weight(choice[0], choice[1]) + G.weight(choice[0], choice[1])
            G_tilde.removeEdge(choice[0], choice[1])
        
        G_tilde.addEdge(choice[0], choice[1], w=new_weight)
    
    if visualize_on:
        visualize(G_tilde)
    return G_tilde


def weight_sampling_sparsification(G, n_samples, visualize_on = False):
    G_tilde = nk.graph.Graph(n=G.numberOfNodes(), weighted=True, edgesIndexed=True)
    G_edge_pmf = np.array([G.weight(u,v) for u,v in G.iterEdges()])*(1/sum([G.weight(u,v) for u,v in G.iterEdges()]))
    edges = np.array([(u,v) for u,v in G.iterEdges()])
    #print(edges[0])
    #print(G_edge_pmf.size)
    
    for sample in range(n_samples):
        #Mercier et al. uses the probability p_e of choosing an edge in the new 
        # weight expression, so we get the index to ensure we can retreive the 
        # right probability from the edge pmf
        choice_index = int(np.random.choice(np.arange(edges.size/2), p=G_edge_pmf))
        #print(choice_index)
        choice = edges[choice_index]
        
        #we ensure that there are no duplicates
        if G_tilde.weight(choice[0], choice[1]) == 0:
            new_weight =(G.weight(choice[0], choice[1]) / (n_samples * G_edge_pmf[choice_index]))
        else:
            new_weight = G_tilde.weight(choice[0], choice[1]) + (G.weight(choice[0], choice[1]) / (n_samples * G_edge_pmf[choice_index]))
            G_tilde.removeEdge(choice[0], choice[1])
        
        G_tilde.addEdge(choice[0], choice[1], w=new_weight)
    
    if visualize_on:
        visualize(G_tilde)
    return G_tilde


def effective_resistance_sampling_sparsification(G, n_samples, visualize_on = False):
    
    # ------------------------------- MODIFIED FROM MERCIER ET AL. PAPER --------------------------------
    # Transform edge list to sparse adj matrix

    def Mtrx_Elist(A):
        #print(A)
        #print(sparse.triu(A))
        triu = sparse.triu(A).toarray()
        j, i = np.nonzero(triu)  # Find edges
        elist = np.vstack((i, j))
        weights = A[triu != 0]  # Find weights
        #print(elist.transpose())
        #print(weights[0, :][0])
        #print(np.array(weights)[0])
        weights = np.array(weights)[0]
        return elist.transpose(), weights

    # Par:
    ## E_list; edge list
    ## weights; edge weights
    def Elist_Mtrx_s(E_list, weights):
        n = np.max(E_list) + 1  # +1 for Python 0-index
        A = sparse.csr_matrix((weights, (E_list[:, 0], E_list[:, 1])), shape=(n, n))
        A = A + A.transpose()

        return A

    # Compute Laplacian, L
    # Par:
    ## A; sparse adj matrix
    def Lap_s(A):
        L = sparse.csgraph.laplacian(A)
        return L

    # Compute signed-edge vertex incidence matrix, B
    # Par:
    ## E_list; edge list
    def sVIM(E_list):
        m = np.shape(E_list)[0]  # number of edges
        E_list = E_list.transpose()  # make rows edge list

        data = [1] * m + [-1] * m  # arbitrary tails and heads
        i = list(range(0, m)) + list(range(0, m))  # i-th positions
        j = E_list[0, :].tolist() + E_list[1, :].tolist()  # j-th positions

        B = sparse.csr_matrix((data, (i, j)))  # Using sparse row matrix format for later use

        return B

    # Compute weights matrix, W
    # Par:
    ## weights; edge weights
    def WDiag(weights):
        m = len(weights)

        weights_sqrt = np.sqrt(weights)  # element-wise sqrt of weights for later use
        W = sparse.dia_matrix((weights_sqrt, [0]), shape=(m, m))  # Use more efficient dia sparse matrix

        return W


    # EffR Approximation
    # method from Koutis et al.
    # Par:
    ## E_list; edge list
    ## weights; list of weights
    ## epsilon; controls accuracy of approximation, increases computation time
    ## method; method of calculation for EffR
    ##
    #### 'ext', exact calculation
    #### 'ssa', original Spielman-Srivastava algorithm
    #### 'kts', Koutis et. al
    ##### Implement preconditioner M for cg solver? cg(A,b,tol,M=None) - 
    #use spilu function or another from scipy.sparse.linalg? 
    #https://stackoverflow.com/questions/32865832/preconditioned-conjugate-gradient-and-linearoperator-in-python
    ##### !Warning! For very small networks, a preconditioner is advised!
    def EffR(E_list, weights, epsilon, method, tol=1e-10):
        # Find number of edges and number of nodes
        m = np.shape(E_list)[0]
        n = np.max(E_list) + 1

        # Obtain necessary matrices from edge list and edge weights
        A = Elist_Mtrx_s(E_list, weights)  # adj matrix - sparse
        L = Lap_s(A)  # Laplacian (sparse array)
        B = sVIM(E_list)  # vertex indices matrix (crs)
        W = WDiag(weights)  # Diagonal weight matrix (dia)
        scale = np.ceil(np.log2(n)) / epsilon  # set scale/resolution for Johnson-Lindenstrauss projection

        M = None

        # Original Spielman-Srivastava algorithm
        if method == 'spl':

            # Define Q in type coo sparse matrix
            Q1 = sparse.random(int(scale), m, 1, format='csr') > 0.5
            Q2 = sparse.random(int(scale), m, 1, format='csr') > 0
            Q_not = Q1 - Q2  # need this to pass by invalid 'not' operator
            Q = Q1 + (-1 * Q_not)  # create Q matrix of 1s and -1s
            Q = Q / np.sqrt(scale)

            SYS = Q @ W @ B  # create system for Johnson-Lindenstrauss projection
            Z = np.zeros(shape=(int(scale), n))  # Create Z matrix to solve smaller dim SYS for effR

            if M is None:  # If no preconditioner
                for i in range(int(scale)):
                    SYSr = SYS[i, :].toarray()
                    Z[i, :] = cg(L, SYSr.transpose(), rtol=tol)[0]  #--------------- ZEE changed tol to RTOL
            else:  # If preconditioner
                for i in range(int(scale)):
                    SYSr = SYS[i, :].toarray()
                    Z[i, :] = cg(L, SYSr.transpose(), rtol=tol, M=M)[0]

            effR = np.sum(np.square(Z[:, E_list[:, 0]] - Z[:, E_list[:, 1]]),
                          axis=0)  # Calculate distance between poitns for effR
            return effR

        # Koutis et al. algorithm
        if method == 'kts':
            effR_res = np.zeros(shape=(1, m))

            if M is None:
                for i in range(int(scale)):
                    ons1 = sparse.random(1, m, 1, format='csr') > 0.5
                    ons2 = sparse.random(1, m, 1, format='csr') > 0
                    ons_not = ons1 - ons2  # need this to pass by invalid 'not' operator
                    ons = ons1 + (-1 * ons_not)  # create Q matrix of 1s and -1s
                    ons = ons / np.sqrt(scale)

                    b = ons @ W @ B
                    b = b.toarray()

                    Z = sparse.linalg.cg(L, b.transpose(), tol=tol)[0]
                    Z = Z.transpose()

                    effR_res = effR_res + np.abs(np.square(Z[E_list[:, 0]] - Z[E_list[:, 1]]))

            else:
                for i in range(int(scale)):
                    # Create memory saving vectors
                    ons1 = sparse.random(1, m, 1, format='csr') > 0.5
                    ons2 = sparse.random(1, m, 1, format='csr') > 0
                    ons_not = ons1 - ons2  # need this to pass by invalid 'not' operator
                    ons = ons1 + (-1 * ons_not)  # create Q matrix of 1s and -1s
                    ons = ons / np.sqrt(scale)

                    b = ons @ W @ B

                    Z = sparse.linalg.cg(L, b.transpose, tol=tol, M=M)[0]
                    Z = Z.transpose()

                    effR_res = effR_res + np.abs(np.square(Z[E_list[:, 0]] - Z[E_list[:, 1]]))

            effR = effR_res[0]
            return effR
    
    # -------------------- END OF CODE MODIFIED FROM MERCIER ET AL. -----------------------
    
    adj_mat = nk.algebraic.adjacencyMatrix(G)
    E_list, weights = Mtrx_Elist(adj_mat)
    
    resistances = EffR(E_list, weights, 0.1, method='spl')
    #print(E_list)
    #print(resistances)
    
    G_tilde = nk.graph.Graph(n=G.numberOfNodes(), weighted=True, edgesIndexed=True)
    #L = nk.algebraic.laplacianMatrix(G)
    #L_plus = scipy.linalg.pinv(L)       #REPLACE WITH APPROXIMATE PSEUDOINVERSE FCN USING SPIELMAN & SRIVASTAVA RANDOM PROJECTION TECHNIQUE
    #dim = L.shape[0]
    
    G_edge_pmf = np.array(resistances)*(1/sum(resistances))
    
    
    for sample in range(n_samples):
        choice_index = int(np.random.choice(np.arange(E_list.size/2), p=G_edge_pmf))
        #print(choice_index)
        edge_to_sample = E_list[choice_index]
        #edge_to_sample = np.random.choice(E_list, p=G_edge_pmf)
        u = edge_to_sample[0]
        v = edge_to_sample[1]
        
        #G_tilde.addEdge(u,v, G.weight(u,v))
        if G_tilde.weight(u, v) == 0:
            new_weight =(G.weight(u, v) / (n_samples * G_edge_pmf[choice_index]))
        else:
            new_weight = G_tilde.weight(u, v) + (G.weight(u, v) / (n_samples * G_edge_pmf[choice_index]))
            G_tilde.removeEdge(u, v)
        
        G_tilde.addEdge(u, v, w=new_weight)
        #x = np.zeros(dim)
        #x[]
        
    #init activity values (infected)
    is_active = G_tilde.attachNodeAttribute("active", int)
    
    #init mu values
    healing_factor = G_tilde.attachNodeAttribute("mu", float)
    components = G_tilde.attachNodeAttribute("components", str)
    for u in G_tilde.iterNodes():
        mu = np.random.random()
        healing_factor[u] = mu
        is_active[u] = 0
        components[u] = f"{u}"
        
    if visualize_on:
        visualize(G_tilde)
    
    return G_tilde
 

def semi_metric_synthetic_network_gen(G, target_tau_metric, viz = False):
    
    # B is the metric backbone
    B = nk.graph.Graph(n=G.numberOfNodes(), weighted = True, edgesIndexed = True)
    
    is_active = B.attachNodeAttribute("active", int)
    healing_factor = B.attachNodeAttribute("mu", float)
    components = B.attachNodeAttribute("components", str)
    
    healing_factor_G = G.getNodeAttribute("mu", float)
    for u in B.iterNodes():
        healing_factor[u] = healing_factor_G[u]
        is_active[u] = 0
        components[u] = f"{u}"
    
    G_dists = copy_graph(G)
    
    E_m = 0
    E_sm = 0
    E = G.numberOfEdges()
    
    #store the semi-metric edges
    E_sm_matrix = np.zeros((E, E))
    
    #store semi-metric distortion values
    distortion_matrix = np.zeros((E, E))
    
    D = np.ones((E, E))*np.inf
    B_matrix = np.zeros((E, E))
    for edge in G.iterEdges():
        d = (1/G.weight(edge[0], edge[1]))-1
        G_dists.setWeight(edge[0], edge[1], d)
        
        #print(f"adding {d} to {edge[0]}, {edge[1]} in D")
        D[edge[0], edge[1]] = d
    
    #all pairs shortest path = APSP    
    APSP_solver = nk.distance.APSP(G_dists)
    APSP_solver.run()
    DTm = APSP_solver.getDistances(asarray=True)
    
    for i in range(E):
        for j in range(E):
            if D[i,j] == DTm[i,j]:
                #add edges to form the metric backbone B, track with B_matrix
                B.addEdge(i,j, w=G.weight(i,j))
                B_matrix[i,j] = G.weight(i,j)
                E_m += 1
            elif D[i,j] != np.inf:
                #keep track of the semi-metric edges 
                E_sm_matrix[i,j] = G.weight(i,j)
                distortion_matrix[i,j] = D[i,j]/DTm[i,j]
                E_sm += 1
                
    backbone_node_list = [u for u in B.iterNodes()]
    tau_metric = E_m/E
    
    if viz:
        visualize(B)
    
    def get_random_nonexistent_edge(G):
        #note that this won't ever stop looking for an edge; if the graph is already complete then it will run forever
        B_np = np.array(backbone_node_list)
        found_valid_edge = False
        
        #this could probably be done in a smarter way, so #TODO make this smarter, especially for big graphs
        edge = np.empty(2)
        while not found_valid_edge:
            u, v = np.random.choice(B_np, replace=False)
            if B.weight(u,v) != 0:
                # we found a metric edge or a semi-metric edge that was alreaddy added; try again
                pass
            else:
                found_valid_edge = True
                edge[0] = u
                edge[1] = v
        
        return edge
                
        
        
    if tau_metric > target_tau_metric:
        print(f"tau_m of the metric backbone ({tau_metric}) is greater than the target tau_m ({target_tau_metric}) and cannot be achieved. Returning metric backbone")
        #print(tau_metric)
    else:
        E_sm_to_add = ((1-tau_metric)/tau_metric)*E_m
        for i in range(E_sm_to_add):
            edge = get_random_nonexistent_edge(B)
            
            #from SMDS paper; bottom part of page 3
            semi_metric_distortion = 1 + np.random.lognormal(mean=0, sigma=1) 
            #from SMDS paper, pgs 10-11
            distance = semi_metric_distortion * DTm[edge[0], edge[1]]
            new_weight = 1/(distance+1)
            
            B.addEdge(edge[0], edge[1], w=new_weight)
    
    return B


def semi_metric_backbone(G, viz = False):
    
    # B is the metric backbone
    B = nk.graph.Graph(n=G.numberOfNodes(), weighted = True, edgesIndexed = True)
    
    is_active = B.attachNodeAttribute("active", int)
    healing_factor = B.attachNodeAttribute("mu", float)
    components = B.attachNodeAttribute("components", str)
    
    healing_factor_G = G.getNodeAttribute("mu", float)
    for u in B.iterNodes():
        healing_factor[u] = healing_factor_G[u]
        is_active[u] = 0
        components[u] = f"{u}"
    G_dists = copy_graph(G)
    
    E = G.numberOfEdges()
    n = G.numberOfNodes()
    
    #store semi-metric distortion values
    D = np.ones((n, n))*np.inf
    
    for edge in G.iterEdges():
        d = (1/G.weight(edge[0], edge[1]))-1
        G_dists.setWeight(edge[0], edge[1], d)
        
        #print(f"adding {d} to {edge[0]}, {edge[1]} in D")
        D[edge[0], edge[1]] = d
    
    #all pairs shortest path = APSP    
    APSP_solver = nk.distance.APSP(G_dists)
    APSP_solver.run()
    DTm = APSP_solver.getDistances(asarray=True)
    
    for i in range(n):
        for j in range(n):
            if D[i,j] == DTm[i,j]:
                #add edges to form the metric backbone B, track with B_matrix
                B.addEdge(i,j, w=G.weight(i,j))
            
    if viz:
        visualize(B)
    
    
    return B


def SMDS_sparsifier(G, chi, output_num_components = False, viz = False):
    
    # B is the metric backbone
    B = nk.graph.Graph(n=G.numberOfNodes(), weighted = True, edgesIndexed = True)
    
    is_active = B.attachNodeAttribute("active", int)
    healing_factor = B.attachNodeAttribute("mu", float)
    components = B.attachNodeAttribute("components", str)
    
    healing_factor_G = G.getNodeAttribute("mu", float)
    for u in B.iterNodes():
        healing_factor[u] = healing_factor_G[u]
        is_active[u] = 0
        components[u] = f"{u}"
    
    G_dists = copy_graph(G)
    
    E_m = 0
    E_sm = 0
    n = G.numberOfNodes()
    
    #store semi-metric distortion values
    semi_metric_edges = set()
    
    D = np.ones((n, n))*np.inf
    
    B_matrix = np.zeros((n, n))
    for edge in G.iterEdges():
        d = (1/G.weight(edge[0], edge[1]))-1
        G_dists.setWeight(edge[0], edge[1], d)
        
        #print(f"adding {d} to {edge[0]}, {edge[1]} in D")
        D[edge[0], edge[1]] = d
    
    #all pairs shortest path = APSP    
    APSP_solver = nk.distance.APSP(G_dists)
    APSP_solver.run()
    DTm = APSP_solver.getDistances(asarray=True)
    
    print("solved APSP")
    
    for i in range(n):
        for j in range(n):
            if D[i,j] == DTm[i,j]:
                #add edges to form the metric backbone B, track with B_matrix
                B.addEdge(i,j, w=G.weight(i,j))
                B_matrix[i,j] = G.weight(i,j)
                E_m += 1
            
            elif D[i,j] != np.inf:
                #keep track of the semi-metric edges 
                semi_metric_distortion = D[i,j]/DTm[i,j]
                semi_metric_edges.add((i, j, semi_metric_distortion))
                
                E_sm += 1
    
    print("Built metric backbone and found all semi-metric edges")
    
    sorted_semi_metric_edges = sorted(list(semi_metric_edges), key=lambda x: x[2])
    
                
    E_sm_to_add = int(round(chi*E_sm))
    for i in range(E_sm_to_add):
        edge = sorted_semi_metric_edges[i]
        B.addEdge(edge[0], edge[1], w=G.weight(edge[0], edge[1]))
        
    if viz:
        visualize(B)
    
    if output_num_components:
        return B, number_of_components(B)
    return B

def get_neil_output(G):
    decimated_sites = []
    for i in range(G.numberOfNodes()):
        G, decimated_sites = sdrg_step(G, neil_mode = False, decimated_sites=decimated_sites, visualizeStep=False)
        
    #print(decimated_sites)
    print("--------------------------")

    temp = [components.split("_") for components in decimated_sites]
    formatted_decimated_sites = [sorted([int(x) for x in sublist]) for sublist in temp]
    print(formatted_decimated_sites)
    

""" ------------------------------------ DCP SIMULATION ------------------------------------ """


def DCP_slow(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0):
    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    if random_start and len(active_nodes) == 0:
        s = np.random.randint(0, len(nodes))
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
        N_active = 1
        active_nodes.add(patient_zero)
        print(patient_zero)
    elif len(active_nodes) == 0:
        is_active[start_node] = 1
        N_active = 1
        active_nodes.add(start_node)
    else:
        N_active = len(active_nodes)
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero))
        
    """
    I'm implementing a modified version of the SIS algorithm described below from 
    (https://journals.aps.org/pre/abstract/10.1103/PhysRevE.86.041125) 
    Epidemic thresholds of the susceptible-infected-susceptible model on 
    networks: A comparison of numerical and theoretical results
    ---------------------------------------------------------------------
    "We consider here the SIS model for epidemics in continuous time. The 
    numerical algorithm is implemented as follows: At each time step, we 
    compute the number of infected nodes, Ni, and links emanating from them, 
    Nn. With probability Ni/(Ni+λNn), one infected node, chosen at random, 
    becomes healthy. With complementary probability λNn/(Ni+λNn), one of the 
    links is selected uniformly at random and the infection is transmitted 
    through it from the infected node corresponding to one of the ends of the 
    edge, toward the (possibly susceptible) node at the other end. The numbers 
    of infected nodes and related links are updated accordingly, time is 
    incremented by Δt=1/(Ni+λNn), and the whole process is iterated."
    
    I've modified this algorithm to account for the disorder in lambda and mu 
    values in our networks; particularly, I take the average lambda value 
    (lambda_avg) of all 'exposed' edges (those with an active site on at least 
                                         one end) 
    for the calculation of whether to heal or infect a site in each timestep. 
    In addition, instead of choosing uniformly among active sites and exposed 
    edges for healing and infection respectively, I construct a probability 
    mass function where the chance of choosing a node or edge is proportional 
    to the mu or lambda value, respectively.  
    
    Some important notes on optimizations:
        - Use sets to store the active nodes, not lists/np arrays. This gives a 
        roughly 100x speed improvement
        - Don't recompute the active nodes each time; the nodes have some attached properties that make this slow
        - It is ok to recompute edges; the computation combines sets (fast) and 
        networkit calls (fast) so it already takes on the order of 10^-5 
        seconds and isn't worth much imrpovement
    
    """
    t = 0
    print("beginning infection")
    
    
    
    while t < t_max:
        #t0=time.time_ns()
        exposed_edges = np.array([
                np.array([u, v]).T 
                for u in active_nodes #outer
                for v in G.iterNeighbors(u) #inner
            ])
        
        
        N_active = len(active_nodes)
        N_exposed = len(exposed_edges)# the number of exposed nodes is the same as the number of edges connected to active sites
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()

        
        if N_active > 0:
            #this is the only time we need active_nodes to be ordered, so we make a list for it
            #active_nodes_list = list(active_nodes)
            #heal_chances = np.array([healing_factor[u] for u in active_nodes_list])
            #convert to pmf
            #heal_pmf = heal_chances*(1/sum(heal_chances))

            prob_to_heal = N_active/(N_active+(lambda_max*N_exposed))
            
            if u1 <= prob_to_heal and not (quasistationary and N_active == 1) and u2 < lambda_max:
                #choose random node to heal?
                site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)
                 
                is_active[site_to_heal] = 0
                
                active_nodes.remove(site_to_heal)
                if track_TOA:
                    TOA_tracker.add((t, site_to_heal))
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
            else:
                
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                exposed_sites = exposed_edges[:, 1]
    
        
                site_to_infect = np.random.choice(exposed_sites)
                is_active[site_to_infect] = 1
                active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                if track_TOA:
                    TOA_tracker.add((t, site_to_infect))
            
            t += 1/(N_active+lambda_max*N_exposed)
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}")
            break
            #t = t_max
        
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%1 == 0:
            print(f"t={t} --- {N_active} nodes are active")
            visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps")
    if track_TOA:
        return TOA_tracker


def DCP(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0):
    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    if random_start and len(active_nodes) == 0:
        s = np.random.randint(0, len(nodes))
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
        N_active = 1
        active_nodes.add(patient_zero)
        print(patient_zero)
    elif len(active_nodes) == 0:
        is_active[start_node] = 1
        N_active = 1
        active_nodes.add(start_node)
    else:
        N_active = len(active_nodes)
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    I'm implementing a modified version of the SIS algorithm described below from 
    (https://journals.aps.org/pre/abstract/10.1103/PhysRevE.86.041125) 
    Epidemic thresholds of the susceptible-infected-susceptible model on 
    networks: A comparison of numerical and theoretical results
    ---------------------------------------------------------------------
    "We consider here the SIS model for epidemics in continuous time. The 
    numerical algorithm is implemented as follows: At each time step, we 
    compute the number of infected nodes, Ni, and links emanating from them, 
    Nn. With probability Ni/(Ni+λNn), one infected node, chosen at random, 
    becomes healthy. With complementary probability λNn/(Ni+λNn), one of the 
    links is selected uniformly at random and the infection is transmitted 
    through it from the infected node corresponding to one of the ends of the 
    edge, toward the (possibly susceptible) node at the other end. The numbers 
    of infected nodes and related links are updated accordingly, time is 
    incremented by Δt=1/(Ni+λNn), and the whole process is iterated."
    
    I've modified this algorithm to account for the disorder in lambda and mu 
    values in our networks; particularly, I take the average lambda value 
    (lambda_avg) of all 'exposed' edges (those with an active site on at least 
                                         one end) 
    for the calculation of whether to heal or infect a site in each timestep. 
    In addition, instead of choosing uniformly among active sites and exposed 
    edges for healing and infection respectively, I construct a probability 
    mass function where the chance of choosing a node or edge is proportional 
    to the mu or lambda value, respectively.  
    
    Some important notes on optimizations:
        - Use sets to store the active nodes, not lists/np arrays. This gives a 
        roughly 100x speed improvement
        - Don't recompute the active nodes each time; the nodes have some attached properties that make this slow
        - Similarly, we don't fully recompute the edges; we keep a single set of exposed edges and 
    """
    t = 0
    print("beginning infection")
    
    exposed_edges = set([
            (u,v) 
            for u in active_nodes #outer
            for v in G.iterNeighbors(u) #inner
        ])
    
    deltaT = 0
    
    while t < t_max:
        #t0=time.time_ns()
        
        N_active = len(active_nodes)
        #print(N_active)
        N_exposed = len(exposed_edges)# the number of exposed nodes is the same as the number of edges connected to active sites
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        u3 = np.random.random()
        
        if N_active > 0:
            #this is the only time we need active_nodes to be ordered, so we make a list for it
            random_site = np.random.choice(list(active_nodes))
            mu_i = healing_factor[random_site]
            
            prob_to_heal = (mu_i*N_active)/((mu_i*N_active)+N_exposed)
            
            if u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)
                 
                is_active[random_site] = 0
                
                newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                for edge in newly_unexposed_neighbors:
                    exposed_edges.remove(edge)
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                    
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
                
                deltaT = 1/(N_active-1)
                t += deltaT
            elif u2 < lambda_max:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                
                exposed_sites = [site for site in G.iterNeighbors(random_site)]
                site_to_infect = np.random.choice(exposed_sites)
                
                if u3 < G.weight(random_site, site_to_infect):
                    #print("infecting")
                    is_active[site_to_infect] = 1
                    active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                    newly_exposed_neighbors = [(site_to_infect,v) 
                            for v in G.iterNeighbors(site_to_infect) #inner
                        ]
                    for edge in newly_exposed_neighbors:
                        exposed_edges.add(edge)
                    if track_TOA:
                        TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
            
                    deltaT = 1/(N_active+1)
                    t += deltaT
            else:
                #print("wasting")
                deltaT = 1/N_active
                t += deltaT
                
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}, the last timestep was {deltaT}")
            break
            #t = t_max
        
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%5000 == 0:
            #pass
            print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker

    
"""
function fast SIS(G,τ, γ, initial infecteds, tmax)
    initialise Q, node statuses and return variables as in fast SIR, but include a source for
    infections with recovery time 0.
    while Q is not empty do
        Event ← earliest remaining event in Q
        if Event.action is transmit then
            if Event.node.status is susceptible then
                process trans SIS(G, Event.node, Event.time, τ, γ, times, S, I, Q, tmax)
            find next trans SIS(Q, Event.source, Event.node, t)  needed for SIS model
        else
            process rec SIS(Event.node, t, S, I)
    return times, S, I

function process trans SIS(G, u, t, τ, γ, times, S, I, Q, tmax)
    append times, S, and I with t, S.last−1, and I.last+1
    u.status ← infected
    u.rec time ← t+exponential variate(γ)
    if u.rec time < tmax then
        newEvent ← {node: u, time: u.rec time, action: recover}
        add newEvent to Q
    for v in G.neighbours(u) do
        find next trans SIS(Q, t, τ, u, v, tmax)
function find next trans SIS(Q, t, τ, source, target, tmax)
    if target.rec time < source.rec time then
        transmission time = max(t, target.rec time)+exponential variate(τ)
        if transmission time < source.rec time then
            newEvent ← {node: target, time: transmission time, action: transmit, source: source}
            push(Q, newEvent)
function process rec SIS(u, times, S, I)
    append times, S, and I with t, S.last+1, and I.last−1
    u.status ← susceptible
"""


def sparsified_DCP(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100):
    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    if random_start and len(active_nodes) == 0:
        s = np.random.randint(0, len(nodes))
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
        N_active = 1
        active_nodes.add(patient_zero)
        print(patient_zero)
    elif len(active_nodes) == 0:
        is_active[start_node] = 1
        N_active = 1
        active_nodes.add(start_node)
    else:
        N_active = len(active_nodes)
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from regular DCP bc it builds the dataframe on the fly as it 
    simulates, and it also has an extra operation to change all of the values 
    in the df which correspond to components of the clusters
    """
    t = 0
    print("beginning infection")
    
    exposed_edges = set([
            (u,v) 
            for u in active_nodes #outer
            for v in G.iterNeighbors(u) #inner
        ])
    
    deltaT = 0
    
    seconds = np.linspace(0, t_max, num=(round(t_max)*original_graph_size)+1)#the graph size is our resolution, so t_max*graph_size gives us the proper timesteps.
    df = pd.DataFrame(False, index=np.arange(original_graph_size), columns=seconds) 
    
    while t < t_max:
        t0=time.time_ns()
        
        N_active = len(active_nodes)
        #print(N_active)
        N_exposed = len(exposed_edges)# the number of exposed nodes is the same as the number of edges connected to active sites
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        u3 = np.random.random()
        
        
        
        if N_active > 0:
            #this is the only time we need active_nodes to be ordered, so we make a list for it
            random_site = np.random.choice(list(active_nodes))
            mu_i = healing_factor[random_site]
            
            prob_to_heal = (mu_i*N_active)/((mu_i*N_active)+N_exposed)
            
            if u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)
                 
                is_active[random_site] = 0
                
                newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                for edge in newly_unexposed_neighbors:
                    exposed_edges.remove(edge)
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                    
                #we could calculate df_col above "if N_active > 0" bc its global, but there are some wasted actions so may as well wait until we know we need it!
                df_col = find_lt(seconds, t)
                for site in components[random_site].split("_"):
                    site_index = int(site)
                    
                    df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                    
                    #df = swap_column_vals_after_k(df, site_index, df_col)
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
                
                deltaT = 1/(N_active-1)
                t += deltaT
            elif u2 < lambda_max:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                
                exposed_sites = [site for site in G.iterNeighbors(random_site)]
                site_to_infect = np.random.choice(exposed_sites)
                
                if u3 < G.weight(random_site, site_to_infect):
                    #print("infecting")
                    is_active[site_to_infect] = 1
                    active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                    newly_exposed_neighbors = [(site_to_infect,v) 
                            for v in G.iterNeighbors(site_to_infect) #inner
                        ]
                    for edge in newly_exposed_neighbors:
                        exposed_edges.add(edge)
                    if track_TOA:
                        TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
                    
                    df_col = find_lt(seconds, t)
                    for site in components[site_to_infect].split("_"):
                        site_index = int(site)
                        df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                        
                        #df = swap_column_vals_after_k(df, site_index, df_col)
                    
                    
                    deltaT = 1/(N_active+1)
                    t += deltaT
            else:
                #print("wasting")
                deltaT = 1/N_active
                t += deltaT
                
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}, the last timestep was {deltaT}")
            break
            #t = t_max
        #print((1/1000000000)*(time.time_ns()-t0))
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%5000 == 0:
            #pass
            print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker
    else:
        return df


def sparsified_DCP_fast(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100):
    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    transition_times = {site : [] for site in range(original_graph_size)}
    
    
    if random_start and len(active_nodes) == 0:
        found_patient_zero_in_largest_component = False
        
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        
        sizes = cc.getComponentSizes()  # Returns a dict: {component_id: size}
        largest_component_id = max(sizes, key=sizes.get)
        s=0
        while not found_patient_zero_in_largest_component:
            s = np.random.randint(0, len(nodes))
            
            node_to_check = s  
            node_component_id = cc.componentOfNode(node_to_check)
        
            found_patient_zero_in_largest_component = (node_component_id == largest_component_id)
        
        
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
        N_active = 1
        
        active_nodes.add(patient_zero)
        #print(patient_zero)
    elif len(active_nodes) == 0:
        #if we give an invalid start node, find the cluster that it's in and make that cluster the start node
        #print(start_node)
        """node_list = [u for u in G.iterNodes()]
        #print(start_node in node_list)
        if start_node not in [u for u in G.iterNodes()]:
            for site in G.iterNodes():
                c = components[site].split("_")
                #print(f"site {site} components are {c}")
                if str(start_node) in c:
                    print(site)
                    start_node = site
        """        
        
        start_node = get_site_location(start_node, G)
            
        is_active[start_node] = 1
        N_active = 1
        #print(start_node)
        active_nodes.add(start_node)
    else:
        N_active = len(active_nodes)
        
    for site in active_nodes:
        #give every node that starts out being active a transition at 0 to represent the initial condition
        #print(site)
        #print(active_nodes)
        #nnn = [components[u] for u in G.iterNodes()]
        #print(nnn)
        #print(components[site])
        c = components[site].split("_")
        
        for component in c:
            #larger node indices appear after sparsification (even though there are fewer nodes), so this prevents out of bounds errors
            transition_times[int(component)].append(0)
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from the original sparsified DCP bc it builds the dataframe 
    differently. The two are not interchangeable because of this different 
    output type
    """
    t = 0
    #print(f"beginning infection")
    
    exposed_edges = set([
            (u,v) 
            for u in active_nodes #outer
            for v in G.iterNeighbors(u) #inner
        ])
    
    deltaT = 0
    
    #the graph size is our resolution, so t_max*graph_size gives us the proper timesteps.
    #seconds = np.linspace(0, t_max, num=(round(t_max)*original_graph_size)+1)
    #need to make sure that the array is large enough to hold 
    
    #df = pd.DataFrame(False, index=np.arange(original_graph_size), columns=seconds) 
    #print(active_nodes)

    #print(exposed_edges)
    
    #for i in active_nodes:
        #print(G.degree(i))

    lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
    while t < t_max:
        #t0=time.time_ns()
        
        N_active = len(active_nodes)
        #print(N_active)
        # the number of exposed nodes is the same as the number of edges connected to active sites
        N_exposed = len(exposed_edges)
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check
        

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        u3 = np.random.random()
        
        
        #t1=time.time_ns()
        #t2=0
        #t3=0
        #t4=0
        #t5=0
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = (mu_i*N_active)/((mu_i*N_active)+N_exposed)
            #t2=time.time_ns()
            #t4=time.time_ns()
            if u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)
                 
                is_active[random_site] = 0
                
                #newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                for neighbor in G.iterNeighbors(random_site):
                    if G.weight(random_site, neighbor) == lambda_max:
                        exposed_edges.remove((random_site, neighbor))
                        if len(exposed_edges) > 0:
                            lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
                        else:
                            lambda_max = 0
                            print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
                            return transition_times
                    else:
                        exposed_edges.remove((random_site, neighbor))
                    
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                    
                #we could calculate df_col above "if N_active > 0" bc its global, but there are some wasted actions so may as well wait until we know we need it!
                #df_col = find_lt(seconds, t)
                for site in components[random_site].split("_"):
                    #site_index = int(site)
                    
                    #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                    
                    transition_times[int(site)].append(t)
                    
                    #df = swap_column_vals_after_k(df, site_index, df_col)
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
                
                deltaT = 1/(N_active-1)
                t += deltaT
                #t5=time.time_ns()
                #print("heal")
            elif u2 < lambda_max:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                if len(list(G.iterNeighbors(random_site))) > 0:
                    exposed_sites = [site for site in G.iterNeighbors(random_site)]
                    site_to_infect = np.random.choice(exposed_sites)
                    
                    if u3 < G.weight(random_site, site_to_infect):
                        #print("infecting")
                        is_active[site_to_infect] = 1
                        active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                        newly_exposed_neighbors = [(site_to_infect,v) 
                                for v in G.iterNeighbors(site_to_infect) #inner
                            ]
                        for edge in newly_exposed_neighbors:
                            exposed_edges.add(edge)
                            newWeight = G.weight(edge[0], edge[1])
                            if newWeight > lambda_max:
                                lambda_max = newWeight
                        if track_TOA:
                            TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
                        
                        #df_col = find_lt(seconds, t)
                        for site in components[site_to_infect].split("_"):
                            #site_index = int(site)
                            #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                            
                            transition_times[int(site)].append(t)
                            #df = swap_column_vals_after_k(df, site_index, df_col)
                        
                        
                    deltaT = 1/(N_active+1)
                    t += deltaT
                    
                    #t5=time.time_ns()
                    #print("inf")
                else:
                    #print("wasting")
                    deltaT = 1/N_active
                    t += deltaT
                    #t5=time.time_ns()
                    #print("waste")
            #t3=time.time_ns()
                
            #print("--")
            #print((t3-t0)/1000000000)
            #print((t5-t4)/1000000000)
            #print((t2-t1)/1000000000)
            
            
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}, the last timestep was {deltaT}")
            break
            #t = t_max
        #print((1/1000000000)*(time.time_ns()-t0))
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%5000 == 0:
            pass
            #print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker
    else:
        return transition_times


def sparsified_DCP_fast_until_percent_infected(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100, percent_infected=0.5):
    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    transition_times = {site : [] for site in range(original_graph_size)}
    
    
    if random_start and len(active_nodes) == 0:
        found_patient_zero_in_largest_component = False
        
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        
        sizes = cc.getComponentSizes()  # Returns a dict: {component_id: size}
        largest_component_id = max(sizes, key=sizes.get)
        s=0
        while not found_patient_zero_in_largest_component:
            s = np.random.randint(0, len(nodes))
            
            node_to_check = s  
            node_component_id = cc.componentOfNode(node_to_check)
        
            found_patient_zero_in_largest_component = (node_component_id == largest_component_id)
        
        
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
        N_active = 1
        
        active_nodes.add(patient_zero)
        #print(patient_zero)
    elif len(active_nodes) == 0:
        #if we give an invalid start node, find the cluster that it's in and make that cluster the start node
        #print(start_node)
        """node_list = [u for u in G.iterNodes()]
        #print(start_node in node_list)
        if start_node not in [u for u in G.iterNodes()]:
            for site in G.iterNodes():
                c = components[site].split("_")
                #print(f"site {site} components are {c}")
                if str(start_node) in c:
                    print(site)
                    start_node = site
        """        
        
        start_node = get_site_location(start_node, G)
            
        is_active[start_node] = 1
        N_active = 1
        #print(start_node)
        active_nodes.add(start_node)
    else:
        N_active = len(active_nodes)
        
    for site in active_nodes:
        #give every node that starts out being active a transition at 0 to represent the initial condition
        #print(site)
        #print(active_nodes)
        #nnn = [components[u] for u in G.iterNodes()]
        #print(nnn)
        #print(components[site])
        c = components[site].split("_")
        
        for component in c:
            #larger node indices appear after sparsification (even though there are fewer nodes), so this prevents out of bounds errors
            transition_times[int(component)].append(0)
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from the original sparsified DCP bc it builds the dataframe 
    differently. The two are not interchangeable because of this different 
    output type
    """
    t = 0
    #print(f"beginning infection")
    
    exposed_edges = set([
            (u,v) 
            for u in active_nodes #outer
            for v in G.iterNeighbors(u) #inner
        ])
    
    deltaT = 0
    
    #the graph size is our resolution, so t_max*graph_size gives us the proper timesteps.
    #seconds = np.linspace(0, t_max, num=(round(t_max)*original_graph_size)+1)
    #need to make sure that the array is large enough to hold 
    
    #df = pd.DataFrame(False, index=np.arange(original_graph_size), columns=seconds) 
    #print(active_nodes)

    #print(exposed_edges)
    
    #for i in active_nodes:
        #print(G.degree(i))

    lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
    while N_active < G.numberOfNodes()*percent_infected:
        #t0=time.time_ns()
        
        N_active = len(active_nodes)
        #print(N_active)
        # the number of exposed nodes is the same as the number of edges connected to active sites
        N_exposed = len(exposed_edges)
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check
        

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        u3 = np.random.random()
        
        
        #t1=time.time_ns()
        #t2=0
        #t3=0
        #t4=0
        #t5=0
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = (mu_i*N_active)/((mu_i*N_active)+N_exposed)
            #t2=time.time_ns()
            #t4=time.time_ns()
            if u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)
                 
                is_active[random_site] = 0
                
                #newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                for neighbor in G.iterNeighbors(random_site):
                    if G.weight(random_site, neighbor) == lambda_max:
                        exposed_edges.remove((random_site, neighbor))
                        if len(exposed_edges) > 0:
                            lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
                        else:
                            lambda_max = 0
                            print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
                            return transition_times
                    else:
                        exposed_edges.remove((random_site, neighbor))
                    
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                    
                #we could calculate df_col above "if N_active > 0" bc its global, but there are some wasted actions so may as well wait until we know we need it!
                #df_col = find_lt(seconds, t)
                for site in components[random_site].split("_"):
                    #site_index = int(site)
                    
                    #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                    
                    transition_times[int(site)].append(t)
                    
                    #df = swap_column_vals_after_k(df, site_index, df_col)
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
                
                deltaT = 1/(N_active-1)
                t += deltaT
                #t5=time.time_ns()
                #print("heal")
            elif u2 < lambda_max:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                if len(list(G.iterNeighbors(random_site))) > 0:
                    exposed_sites = [site for site in G.iterNeighbors(random_site)]
                    site_to_infect = np.random.choice(exposed_sites)
                    
                    if u3 < G.weight(random_site, site_to_infect):
                        #print("infecting")
                        is_active[site_to_infect] = 1
                        active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                        newly_exposed_neighbors = [(site_to_infect,v) 
                                for v in G.iterNeighbors(site_to_infect) #inner
                            ]
                        for edge in newly_exposed_neighbors:
                            exposed_edges.add(edge)
                            newWeight = G.weight(edge[0], edge[1])
                            if newWeight > lambda_max:
                                lambda_max = newWeight
                        if track_TOA:
                            TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
                        
                        #df_col = find_lt(seconds, t)
                        for site in components[site_to_infect].split("_"):
                            #site_index = int(site)
                            #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                            
                            transition_times[int(site)].append(t)
                            #df = swap_column_vals_after_k(df, site_index, df_col)
                        
                        
                    deltaT = 1/(N_active+1)
                    t += deltaT
                    
                    #t5=time.time_ns()
                    #print("inf")
                else:
                    #print("wasting")
                    deltaT = 1/N_active
                    t += deltaT
                    #t5=time.time_ns()
                    #print("waste")
            #t3=time.time_ns()
                
            #print("--")
            #print((t3-t0)/1000000000)
            #print((t5-t4)/1000000000)
            #print((t2-t1)/1000000000)
            
            
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}, the last timestep was {deltaT}")
            break
            #t = t_max
        #print((1/1000000000)*(time.time_ns()-t0))
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%5000 == 0:
            pass
            #print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    return t


def sparsified_DCP_SI(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100):
    patient_zero = 0
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    transition_times = {site : [] for site in range(original_graph_size)}
    
    
    if random_start and len(active_nodes) == 0:
        found_patient_zero_in_largest_component = False
        
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        
        sizes = cc.getComponentSizes()  # Returns a dict: {component_id: size}
        largest_component_id = max(sizes, key=sizes.get)
        s=0
        while not found_patient_zero_in_largest_component:
            s = np.random.randint(0, len(nodes))
            
            node_to_check = s  
            node_component_id = cc.componentOfNode(node_to_check)
        
            found_patient_zero_in_largest_component = (node_component_id == largest_component_id)
        
        
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
        N_active = 1
        
        active_nodes.add(patient_zero)
        #print(patient_zero)
    elif len(active_nodes) == 0:
        #if we give an invalid start node, find the cluster that it's in and make that cluster the start node
        #print(start_node)
        """node_list = [u for u in G.iterNodes()]
        #print(start_node in node_list)
        if start_node not in [u for u in G.iterNodes()]:
            for site in G.iterNodes():
                c = components[site].split("_")
                #print(f"site {site} components are {c}")
                if str(start_node) in c:
                    print(site)
                    start_node = site
        """        
        
        start_node = get_site_location(start_node, G)
            
        is_active[start_node] = 1
        N_active = 1
        #print(start_node)
        active_nodes.add(start_node)
    else:
        N_active = len(active_nodes)
        
    for site in active_nodes:
        #give every node that starts out being active a transition at 0 to represent the initial condition
        #print(site)
        #print(active_nodes)
        #nnn = [components[u] for u in G.iterNodes()]
        #print(nnn)
        #print(components[site])
        c = components[site].split("_")
        
        for component in c:
            #larger node indices appear after sparsification (even though there are fewer nodes), so this prevents out of bounds errors
            transition_times[int(component)].append(0)
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from the original sparsified DCP bc it builds the dataframe 
    differently. The two are not interchangeable because of this different 
    output type
    """
    t = 0
    #print(f"beginning infection")
    
    exposed_edges = set([
            (u,v) 
            for u in active_nodes #outer
            for v in G.iterNeighbors(u) #inner
        ])
    
    deltaT = 0
    

    while t < t_max:
        #t0=time.time_ns()
        
        N_active = len(active_nodes)

        u1 = np.random.random()
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it

            if len(list(G.iterNeighbors(random_site))) > 0:
                exposed_sites = [site for site in G.iterNeighbors(random_site)]
                site_to_infect = np.random.choice(exposed_sites)
                
                if u1 < G.weight(random_site, site_to_infect):
                    #print("infecting")
                    is_active[site_to_infect] = 1
                    active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                    newly_exposed_neighbors = [(site_to_infect,v) 
                            for v in G.iterNeighbors(site_to_infect) #inner
                        ]
                    for edge in newly_exposed_neighbors:
                        exposed_edges.add(edge)
                        
                    if track_TOA:
                        TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
                    
                    #df_col = find_lt(seconds, t)
                    for site in components[site_to_infect].split("_"):
                        #site_index = int(site)
                        #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                        
                        transition_times[int(site)].append(t)
                        #df = swap_column_vals_after_k(df, site_index, df_col)
                    
                    
                deltaT = 1/(N_active+1)
                t += deltaT
                
                #t5=time.time_ns()
                #print("inf")
            else:
                #print("wasting")
                deltaT = 1/N_active
                t += deltaT
                #t5=time.time_ns()
                #print("waste")
                
            #print("--")
            #print((t3-t0)/1000000000)
            #print((t5-t4)/1000000000)
            #print((t2-t1)/1000000000)
            
            
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}, the last timestep was {deltaT}")
            break
            #t = t_max
        #print((1/1000000000)*(time.time_ns()-t0))
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%5000 == 0:
            pass
            #print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker
    else:
        return transition_times



def sparsified_DCP_fast_SIR_old(G, track_TOA=False, quasistationary = False, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100):
    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    transition_times = {site : [] for site in range(original_graph_size)}
    
    
    if random_start and len(active_nodes) == 0:
        found_patient_zero_in_largest_component = False
        
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        
        sizes = cc.getComponentSizes()  # Returns a dict: {component_id: size}
        largest_component_id = max(sizes, key=sizes.get)
        s=0
        while not found_patient_zero_in_largest_component:
            s = np.random.randint(0, len(nodes))
            
            node_to_check = s  
            node_component_id = cc.componentOfNode(node_to_check)
        
            found_patient_zero_in_largest_component = (node_component_id == largest_component_id)
        
        
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
        N_active = 1
        
        active_nodes.add(patient_zero)
        #print(patient_zero)
    elif len(active_nodes) == 0:
        #if we give an invalid start node, find the cluster that it's in and make that cluster the start node
        #print(start_node)
        """node_list = [u for u in G.iterNodes()]
        #print(start_node in node_list)
        if start_node not in [u for u in G.iterNodes()]:
            for site in G.iterNodes():
                c = components[site].split("_")
                #print(f"site {site} components are {c}")
                if str(start_node) in c:
                    print(site)
                    start_node = site
        """        
        
        start_node = get_site_location(start_node, G)
            
        is_active[start_node] = 1
        N_active = 1
        #print(start_node)
        active_nodes.add(start_node)
    else:
        N_active = len(active_nodes)
        
    for site in active_nodes:
        #give every node that starts out being active a transition at 0 to represent the initial condition
        #print(site)
        #print(active_nodes)
        #nnn = [components[u] for u in G.iterNodes()]
        #print(nnn)
        #print(components[site])
        c = components[site].split("_")
        
        for component in c:
            #larger node indices appear after sparsification (even though there are fewer nodes), so this prevents out of bounds errors
            transition_times[int(component)].append(0)
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from the original sparsified DCP bc it builds the dataframe 
    differently. The two are not interchangeable because of this different 
    output type
    """
    t = 0
    #print(f"beginning infection")
    
    exposed_edges = set([
            (u,v) 
            for u in active_nodes #outer
            for v in G.iterNeighbors(u) #inner
        ])
    
    deltaT = 0
    
    #the graph size is our resolution, so t_max*graph_size gives us the proper timesteps.
    #seconds = np.linspace(0, t_max, num=(round(t_max)*original_graph_size)+1)
    #need to make sure that the array is large enough to hold 
    
    #df = pd.DataFrame(False, index=np.arange(original_graph_size), columns=seconds) 
    #print(active_nodes)

    #print(exposed_edges)
    
    #for i in active_nodes:
        #print(G.degree(i))

    lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
    
    mu_list = [healing_factor[u] for u in G.iterNodes()]
    
    num_small_mu = 0
    #some mu values are so small that they will almost never occur. We choose 10^-4
    for i in range(len(mu_list)):
        if mu_list[i] < 0.0001:
            num_small_mu += 1
    while t < t_max:
        #t0=time.time_ns()
        
        N_active = len(active_nodes)
        #print(N_active)
        # the number of exposed nodes is the same as the number of edges connected to active sites
        N_exposed = len(exposed_edges)
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check
        

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        u3 = np.random.random()
        
        
        #t1=time.time_ns()
        #t2=0
        #t3=0
        #t4=0
        #t5=0
        
        

        if N_active > 0:
            
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = (mu_i*N_active)/((mu_i*N_active)+N_exposed)
            #t2=time.time_ns()
            #t4=time.time_ns()
            
            if N_active < num_small_mu:
                exposed_edge_recoveries = [is_active[u[1]] for u in exposed_edges]
                #print(exposed_edge_recoveries)
                if sum(exposed_edge_recoveries) == 2*len(exposed_edge_recoveries):
                    u1=0
            
            if u1 <= prob_to_heal:
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)
                 
                is_active[random_site] = 2 #2 corresponds to R (Recovered) in SIR
                #print(f"healed site {random_site}")
                #newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                for neighbor in G.iterNeighbors(random_site):
                    if G.weight(random_site, neighbor) == lambda_max:
                        exposed_edges.remove((random_site, neighbor))
                        if len(exposed_edges) > 0:
                            lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
                        else:
                            lambda_max = 0
                            print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
                            return transition_times
                    else:
                        exposed_edges.remove((random_site, neighbor))
                    
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                    
                #we could calculate df_col above "if N_active > 0" bc its global, but there are some wasted actions so may as well wait until we know we need it!
                #df_col = find_lt(seconds, t)
                for site in components[random_site].split("_"):
                    #site_index = int(site)
                    
                    #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                    
                    transition_times[int(site)].append(t)
                    
                    #df = swap_column_vals_after_k(df, site_index, df_col)
                #= active_nodes[active_nodes!=site_to_heal]
                
                
                deltaT = 1/(N_active-1)
                t += deltaT
                #t5=time.time_ns()
                #print("heal")
            elif u2 < lambda_max:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                if len(list(G.iterNeighbors(random_site))) > 0:
                    exposed_sites = [site for site in G.iterNeighbors(random_site)]
                    site_to_infect = np.random.choice(exposed_sites)
                    
                    if u3 < G.weight(random_site, site_to_infect) and is_active[site_to_infect] == 0:
                        #print("infecting")
                        is_active[site_to_infect] = 1
                        #print(f"infected site {site_to_infect}")
                        active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                        newly_exposed_neighbors = [(site_to_infect,v) 
                                for v in G.iterNeighbors(site_to_infect) #inner
                            ]
                        for edge in newly_exposed_neighbors:
                            exposed_edges.add(edge)
                            newWeight = G.weight(edge[0], edge[1])
                            if newWeight > lambda_max:
                                lambda_max = newWeight
                        if track_TOA:
                            TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
                        
                        #df_col = find_lt(seconds, t)
                        for site in components[site_to_infect].split("_"):
                            #site_index = int(site)
                            #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                            
                            transition_times[int(site)].append(t)
                            #df = swap_column_vals_after_k(df, site_index, df_col)
                        
                        
                    deltaT = 1/(N_active+1)
                    t += deltaT
                    
                    #t5=time.time_ns()
                    #print("inf")
                else:
                    #print("wasting")
                    
                    deltaT = 1/N_active
                    t += deltaT
                    #t5=time.time_ns()
                    #print("waste")
            #t3=time.time_ns()
                
            #print("--")
            #print((t3-t0)/1000000000)
            #print((t5-t4)/1000000000)
            #print((t2-t1)/1000000000)
            
            
            step_counter+=1
            if step_counter%5000 == 0:
                #print(f"t={t} --- {N_active} nodes are active")
                #print(mu_i)
                #print(N_exposed)
                #a = [1 if is_active[u] == 2 else 0 for u in G.iterNodes()]
                #print(f"{sum(a)} nodes are recovered")
                pass
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}, the last timestep was {deltaT}")
            break
            #t = t_max
        #print((1/1000000000)*(time.time_ns()-t0))
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        
            
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker
    else:
        return transition_times


def sparsified_DCP_fast_SIR(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100):
    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    transition_times = {site : [] for site in range(original_graph_size)}
    
    
    if random_start and len(active_nodes) == 0:
        found_patient_zero_in_largest_component = False
        
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        
        sizes = cc.getComponentSizes()  # Returns a dict: {component_id: size}
        largest_component_id = max(sizes, key=sizes.get)
        s=0
        while not found_patient_zero_in_largest_component:
            s = np.random.randint(0, len(nodes))
            
            node_to_check = s  
            node_component_id = cc.componentOfNode(node_to_check)
        
            found_patient_zero_in_largest_component = (node_component_id == largest_component_id)
        
        
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
        N_active = 1
        
        active_nodes.add(patient_zero)
        #print(patient_zero)
    elif len(active_nodes) == 0:
        start_node = get_site_location(start_node, G)
            
        is_active[start_node] = 1
        N_active = 1
        #print(start_node)
        active_nodes.add(start_node)
    else:
        N_active = len(active_nodes)
        
    for site in active_nodes:
        #give every node that starts out being active a transition at 0 to represent the initial condition
        
        c = components[site].split("_")
        
        for component in c:
            #larger node indices appear after sparsification (even though there are fewer nodes), so this prevents out of bounds errors
            transition_times[int(component)].append(0)
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from the original sparsified DCP bc it builds the dataframe 
    differently. The two are not interchangeable because of this different 
    output type
    """
    t = 0    
    exposed_edges = set([
            (u,v) 
            for u in active_nodes #outer
            for v in G.iterNeighbors(u) #inner
        ])
    
    deltaT = 0

    lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
    while t < t_max:
        #t0=time.time_ns()
        
        N_active = len(active_nodes)
        N_exposed = len(exposed_edges)
               

        u1 = np.random.random()
        u2 = np.random.random()
        u3 = np.random.random()
        
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = (mu_i*N_active)/((mu_i*N_active)+N_exposed)
            
            if u1 <= prob_to_heal:
    
                is_active[random_site] = 2
    
                active_nodes.remove(random_site)
    
                # Remove all exposed edges originating from this node
                for neighbor in G.iterNeighbors(random_site):
                    if G.weight(random_site, neighbor) == lambda_max:
                        exposed_edges.remove((random_site, neighbor))
                        lambda_max = max([G.weight(u, v) for u, v in exposed_edges])
                    else:
                        print(exposed_edges)
                        exposed_edges.remove((random_site, neighbor))
    
                """# Other infected neighbors can no longer infect this recovered node
                for neighbor in G.iterNeighbors(random_site):
                    exposed_edges.remove((neighbor, random_site))"""
                
                if not len(exposed_edges):
                    lambda_max = 0
    
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
    
                for site in components[random_site].split("_"):
                    transition_times[int(site)].append(t)
    
                if len(active_nodes) > 0:
                    deltaT = 1 / len(active_nodes)
                    t += deltaT
    
            elif u2 < lambda_max:
    
                neighbors = [v for v in G.iterNeighbors(random_site) if is_active[v] == 0]
    
                if len(neighbors):
    
                    site_to_infect = np.random.choice(neighbors)
    
                    if u3 < G.weight(random_site, site_to_infect):
    
                        is_active[site_to_infect] = 1
                        active_nodes.add(site_to_infect)
    
                        if track_TOA:
                            TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
    
                        for site in components[site_to_infect].split("_"):
                            transition_times[int(site)].append(t)
    
                        # New infected node exposes susceptible neighbors
                        for neighbor in G.iterNeighbors(site_to_infect):
                            if is_active[neighbor] == 0:
                                exposed_edges.add(
                                    (site_to_infect, neighbor)
                                )
    
                                w = G.weight(site_to_infect, neighbor)
                                if w > lambda_max:
                                    lambda_max = w
    
                        """# Newly infected node is no longer susceptible
                        for nbr in G.iterNeighbors(site_to_infect):
                            exposed_edges.discard((nbr, site_to_infect))"""
    
                    deltaT = 1 / len(active_nodes)
                    t += deltaT
    
                else:
    
                    deltaT = 1 / len(active_nodes)
                    t += deltaT
    
            step_counter += 1
            """
            if u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                 
                is_active[random_site] = 0
                
                for neighbor in G.iterNeighbors(random_site):
                    if G.weight(random_site, neighbor) == lambda_max:
                        exposed_edges.remove((random_site, neighbor))
                        if len(exposed_edges) > 0:
                            lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
                        else:
                            lambda_max = 0
                            print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
                            return transition_times
                    else:
                        exposed_edges.remove((random_site, neighbor))
                    
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                   
                for site in components[random_site].split("_"):
                    
                    transition_times[int(site)].append(t)
                    
                deltaT = 1/(N_active-1)
                t += deltaT
                
            elif u2 < lambda_max:
                
                if len(list(G.iterNeighbors(random_site))) > 0:
                    exposed_sites = [site for site in G.iterNeighbors(random_site)]
                    site_to_infect = np.random.choice(exposed_sites)
                    
                    if u3 < G.weight(random_site, site_to_infect):
                        is_active[site_to_infect] = 1
                        active_nodes.add(site_to_infect)
                        newly_exposed_neighbors = [(site_to_infect,v) 
                                for v in G.iterNeighbors(site_to_infect) #inner
                            ]
                        for edge in newly_exposed_neighbors:
                            exposed_edges.add(edge)
                            newWeight = G.weight(edge[0], edge[1])
                            if newWeight > lambda_max:
                                lambda_max = newWeight
                        if track_TOA:
                            TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
                        
                        for site in components[site_to_infect].split("_"):
                            
                            transition_times[int(site)].append(t)
                        
                        
                    deltaT = 1/(N_active+1)
                    t += deltaT
                    
                else:
                    deltaT = 1/N_active
                    t += deltaT
                    
            
            step_counter+=1
            """
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}, the last timestep was {deltaT}")
            break
            
        if step_counter%5000 == 0:
            pass
            #print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker
    else:
        return transition_times



def sparsified_DCP_fast_SIR_steps(G, track_TOA=False, quasistationary = True, steps_max = 1000, random_start = True, start_node = 0, original_graph_size = 100):
    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    transition_times = {site : [] for site in range(original_graph_size)}
    
    
    if random_start and len(active_nodes) == 0:
        found_patient_zero_in_largest_component = False
        
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        
        sizes = cc.getComponentSizes()  # Returns a dict: {component_id: size}
        largest_component_id = max(sizes, key=sizes.get)
        s=0
        while not found_patient_zero_in_largest_component:
            s = np.random.randint(0, len(nodes))
            
            node_to_check = s  
            node_component_id = cc.componentOfNode(node_to_check)
        
            found_patient_zero_in_largest_component = (node_component_id == largest_component_id)
        
        
        patient_zero = nodes[s]
        is_active[patient_zero] = 1
        N_active = 1
        
        active_nodes.add(patient_zero)
        #print(patient_zero)
    elif len(active_nodes) == 0:
        #if we give an invalid start node, find the cluster that it's in and make that cluster the start node
        #print(start_node)
        """node_list = [u for u in G.iterNodes()]
        #print(start_node in node_list)
        if start_node not in [u for u in G.iterNodes()]:
            for site in G.iterNodes():
                c = components[site].split("_")
                #print(f"site {site} components are {c}")
                if str(start_node) in c:
                    print(site)
                    start_node = site
        """        
        
        start_node = get_site_location(start_node, G)
            
        is_active[start_node] = 1
        N_active = 1
        #print(start_node)
        active_nodes.add(start_node)
    else:
        N_active = len(active_nodes)
        
    for site in active_nodes:
        #give every node that starts out being active a transition at 0 to represent the initial condition
        #print(site)
        #print(active_nodes)
        #nnn = [components[u] for u in G.iterNodes()]
        #print(nnn)
        #print(components[site])
        c = components[site].split("_")
        
        for component in c:
            #larger node indices appear after sparsification (even though there are fewer nodes), so this prevents out of bounds errors
            transition_times[int(component)].append(0)
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from the original sparsified DCP bc it builds the dataframe 
    differently. The two are not interchangeable because of this different 
    output type
    """
    t = 0
    #print(f"beginning infection")
    
    exposed_edges = set([
            (u,v) 
            for u in active_nodes #outer
            for v in G.iterNeighbors(u) #inner
        ])
    
    deltaT = 0
    
    #the graph size is our resolution, so t_max*graph_size gives us the proper timesteps.
    #seconds = np.linspace(0, t_max, num=(round(t_max)*original_graph_size)+1)
    #need to make sure that the array is large enough to hold 
    
    #df = pd.DataFrame(False, index=np.arange(original_graph_size), columns=seconds) 
    #print(active_nodes)

    #print(exposed_edges)
    
    #for i in active_nodes:
        #print(G.degree(i))

    lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
    while step_counter < steps_max:
        #t0=time.time_ns()
        
        N_active = len(active_nodes)
        #print(N_active)
        # the number of exposed nodes is the same as the number of edges connected to active sites
        N_exposed = len(exposed_edges)
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check
        

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        u3 = np.random.random()
        
        
        #t1=time.time_ns()
        #t2=0
        #t3=0
        #t4=0
        #t5=0
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = (mu_i*N_active)/((mu_i*N_active)+N_exposed)
            #t2=time.time_ns()
            #t4=time.time_ns()
            if u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)
                 
                is_active[random_site] = 2 #2 corresponds to R (Recovered) in SIR
                
                #newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                for neighbor in G.iterNeighbors(random_site):
                    if G.weight(random_site, neighbor) == lambda_max:
                        exposed_edges.remove((random_site, neighbor))
                        if len(exposed_edges) > 0:
                            lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
                        else:
                            lambda_max = 0
                            print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
                            return transition_times
                    else:
                        exposed_edges.remove((random_site, neighbor))
                    
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                    
                #we could calculate df_col above "if N_active > 0" bc its global, but there are some wasted actions so may as well wait until we know we need it!
                #df_col = find_lt(seconds, t)
                for site in components[random_site].split("_"):
                    #site_index = int(site)
                    
                    #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                    
                    transition_times[int(site)].append(t)
                    
                    #df = swap_column_vals_after_k(df, site_index, df_col)
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
                
                deltaT = 1/(N_active-1)
                t += deltaT
                #t5=time.time_ns()
                #print("heal")
            elif u2 < lambda_max:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                if len(list(G.iterNeighbors(random_site))) > 0:
                    exposed_sites = [site for site in G.iterNeighbors(random_site)]
                    site_to_infect = np.random.choice(exposed_sites)
                    
                    if u3 < G.weight(random_site, site_to_infect) and is_active[site_to_infect] == 0:
                        #print("infecting")
                        is_active[site_to_infect] = 1
                        active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                        newly_exposed_neighbors = [(site_to_infect,v) 
                                for v in G.iterNeighbors(site_to_infect) #inner
                            ]
                        for edge in newly_exposed_neighbors:
                            exposed_edges.add(edge)
                            newWeight = G.weight(edge[0], edge[1])
                            if newWeight > lambda_max:
                                lambda_max = newWeight
                        if track_TOA:
                            TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
                        
                        #df_col = find_lt(seconds, t)
                        for site in components[site_to_infect].split("_"):
                            #site_index = int(site)
                            #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                            
                            transition_times[int(site)].append(t)
                            #df = swap_column_vals_after_k(df, site_index, df_col)
                        
                        
                    deltaT = 1/(N_active+1)
                    t += deltaT
                    
                    #t5=time.time_ns()
                    #print("inf")
                else:
                    #print("wasting")
                    deltaT = 1/N_active
                    t += deltaT
                    #t5=time.time_ns()
                    #print("waste")
            #t3=time.time_ns()
                
            #print("--")
            #print((t3-t0)/1000000000)
            #print((t5-t4)/1000000000)
            #print((t2-t1)/1000000000)
            
            
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}, the last timestep was {deltaT}")
            break
            #t = t_max
        #print((1/1000000000)*(time.time_ns()-t0))
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%5000 == 0:
            pass
            #print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker
    else:
        return transition_times



def estimate_runtime(graph_size, t_max, avg_proportion_infected=0.7):
    #0.00015 is a rough estimate of how long each timestep takes
    s = graph_size*t_max*avg_proportion_infected*0.00015
    print(f"{s} seconds, or")
    print(f"{s/60} minutes, or")
    print(f"{s/(60*60)} hrs")
    return 



def reset_DCP(G):    
    is_active = G.getNodeAttribute("active", int)
    for u in G.iterNodes():
        is_active[u] = 0
        
def reset_to_distributed_infection(G, proportion):
    is_active = G.getNodeAttribute("active", int)
    for u in G.iterNodes():
        x = np.random.random()
        if x < proportion:
            is_active[u] = 1
        else:
            is_active[u] = 0


""" ------------------------------------ COLLECT DCP DATA ------------------------------------ """


def get_DCP_data(G, t_max, n_sims):
    
    resolution = 10 #timestep for tracking is 1/resolution. Use higher resolutions for bigger graphs
    
    reset_DCP(G)
    simulation_data = []
    for i in range(n_sims):
        node_state_change_set = DCP(G, track_TOA=True, t_max=t_max, random_start=False)
        print(node_state_change_set)
        #df = pd.DataFrame
        seconds = np.linspace(0, t_max, num=(round(t_max)*resolution)+1)
        df = pd.DataFrame(0, index=np.arange(G.numberOfNodes()), columns=seconds) 
        for i in range(0,t_max*resolution):
            t = i/resolution
            events_to_remove = []
            for state_change in node_state_change_set:
                if state_change[0] < t:
                    df = swap_column_vals_after_k(df, state_change[1], i)
                    events_to_remove.append(state_change)
            for state_change in events_to_remove:
                
                node_state_change_set.remove(state_change)
                    
                    #df.iat[state_change[1], i] = 
        simulation_data.append(df)
        reset_DCP(G)
    
    return simulation_data


def get_DCP_data_asymptotic(G, t_max, resolution):
    """
    For situations where we want to track the asymptotic behavior of DCP on a 
    graph, so we simulate for a relaxation period without tracking anything, 
    *then* begin to track only after relaxation. This function only tracks the 
    things after the relaxation period.
    """
    #timestep for tracking is 1/resolution. Use higher resolutions for bigger graphs
    #A good resolution is usually G.numberOfNodes + 1; this guarantees that you won't get "simultaneous" events that appear to happen in the exact same timestep
    #You could just use G.numberOfNodes though unless a time will come where every site is infected
    node_state_change_set = DCP(G, track_TOA=True, t_max=t_max, quasistationary=True, random_start=True)
    #print(node_state_change_set)
    #df = pd.DataFrame
    seconds = np.linspace(0, t_max, num=(round(t_max)*resolution)+1)
    df = pd.DataFrame(False, index=np.arange(G.numberOfNodes()), columns=seconds) 
    
    #j = 1
    for state_change in node_state_change_set:
        nearest_time_index = find_lt(seconds, state_change[0])
        #print(state_change[0])
        #print(nearest_time)
        #df.iloc[state_change[1], nearest_time_index:] = 99
        df.iloc[state_change[1], nearest_time_index:] = np.where(df.iloc[state_change[1], nearest_time_index:] == True, False, True)
        #swap_column_vals_after_k(df, state_change[1], nearest_time_index)
        #print(f"{j}/{len(node_state_change_set)}")
        #j+=1
            
    return df


def find_lt(arr, x):
    'Find index of rightmost value less than x. Copied from bisect docs and modified'
    i = bisect.bisect_left(arr, x)
    """if i == 0:
        return 0"""
    return i-1
    

def asymptotic_quasistationary_activity_probability(G, G_structure="chain"):
    relaxation_time = 5000
    DCP(G, t_max=relaxation_time, quasistationary=True, random_start=True)
    data = get_DCP_data_asymptotic(G, relaxation_time, G.numberOfNodes())
    
    matrix = np.matrix(data.to_numpy())
    
    #print(matrix.shape)

    proportions_of_time_active_listform = matrix.sum(axis=1)
    proportions_of_time_active = [u[0,0] for u in proportions_of_time_active_listform]
    
    #print(proportions_of_time_active)
    if G_structure == "chain":
        plt.bar(range(G.numberOfNodes()), proportions_of_time_active, width=1, linewidth=0)
    if G_structure == "lattice":

        Z = np.random.rand(6, 10)
        x = np.arange(-0.5, 10, 1)  # len = 11
        y = np.arange(4.5, 11, 1)  # len = 7
        
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, Z)
        
        fig.tight_layout()
        plt.show()
    
    #data.to_excel("output_{time.time_ns()}.xlsx")
    return data


def sparsified_asymptotic_quasistationary_activity_probability(G, G_structure="chain", original_graph_size=100, dimensions = [1, 1]):
    relaxation_time = 25
    DCP(G, t_max=relaxation_time, quasistationary=True, random_start=True)
    data = sparsified_DCP(G, t_max=relaxation_time, original_graph_size=original_graph_size)
    
    matrix = np.matrix(data.to_numpy())
    
    #print(matrix.shape)

    proportions_of_time_active_listform = matrix.sum(axis=1)
    proportions_of_time_active = [u[0,0] for u in proportions_of_time_active_listform]
    
    #print(proportions_of_time_active)
    if G_structure == "chain":
        plt.bar(range(original_graph_size), proportions_of_time_active, width=1, linewidth=0)
    if G_structure == "lattice":
        # same code as vis_lat
        h = dimensions[0]
        w = dimensions[1]
        
        Z = np.zeros((h,w))
        
        #first build a bunch of lines/rows
        for y in range(h):
            for x in range(w):  
                Z[x,y] = proportions_of_time_active[x+(y*w)]
        
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, Z, cmap='viridis')
        fig.tight_layout()

        #fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
        
        # plot just the positive data and save the
        # color "mappable" object returned by ax1.imshow
        pos = ax.imshow(Z, cmap='viridis', interpolation='none')
        
        # add the colorbar using the figure's method,
        # telling which mappable we're talking about and
        # which Axes object it should be near
        fig.colorbar(pos, ax=ax)    
        plt.show()
    #data.to_excel("output_{time.time_ns()}.xlsx")
    return data


def fast_sparsified_asymptotic_quasistationary_activity_probability(G, t_relax = 30, G_structure="chain", original_graph_size=100, dimensions = [1, 1], viz=True):
    relaxation_time = t_relax
    sparsified_DCP_fast(G, t_max=relaxation_time, original_graph_size=original_graph_size)
    #data is a dictionary with keys = sites and values = list of transition times
    data = sparsified_DCP_fast(G, t_max=relaxation_time, original_graph_size=original_graph_size)
    
    
    if viz:
        #matrix = np.matrix(data.to_numpy())
        
        #print(matrix.shape)
        total_time_active_list = np.empty(original_graph_size)
        for site in data:
            trans_times = data[site]
            total_time_active = 0
            for i in range(1, len(trans_times), 2):
                total_time_active += trans_times[i] - trans_times[i-1]
            total_time_active_list[site] = total_time_active
        
        proportions_of_time_active = total_time_active_list/relaxation_time
        #print(proportions_of_time_active_listform)
        #proportions_of_time_active = [u[0,0] for u in proportions_of_time_active_listform]
        #print(proportions_of_time_active)
        #print(proportions_of_time_active)
        if G_structure == "chain":
            plt.bar(range(original_graph_size), proportions_of_time_active, width=1, linewidth=0)
        if G_structure == "lattice":
            # same code as vis_lat
            h = dimensions[0]
            w = dimensions[1]
            
            Z = np.zeros((h,w))
            
            #first build a bunch of lines/rows
            for y in range(h):
                for x in range(w):  
                    Z[x,y] = proportions_of_time_active[x+(y*w)]
            
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)
            
            fig, ax = plt.subplots()
            ax.pcolormesh(x, y, Z, cmap='viridis')
            fig.tight_layout()
    
            #fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
            
            # plot just the positive data and save the
            # color "mappable" object returned by ax1.imshow
            pos = ax.imshow(Z, cmap='viridis', interpolation='none')
            
            # add the colorbar using the figure's method,
            # telling which mappable we're talking about and
            # which Axes object it should be near
            fig.colorbar(pos, ax=ax)    
            fig.title(f"{100*(1-(G.numberOfNodes()/original_graph_size))}% sparsified", fontsize=12)
            plt.show()

    return data



""" ------------------------------------ QUANTIFY SPARSIFIER QUALITY ------------------------------------ """

def t_half(G, sim_time, num_simulations=10, dispersion_density=0.01, p=0.5, original_graph_size=100):
    reset_DCP(G)
    
        
    half_times = np.empty(num_simulations)
        
    for i in range(num_simulations):
        t_half_sim = sparsified_DCP_fast_until_percent_infected(G, t_max = sim_time, original_graph_size=original_graph_size, percent_infected=p)
        half_times[i] = t_half_sim
        reset_DCP(G)
    
    return np.average(half_times)

def xi_half_B(G, sim_time, num_simulations=10):
    """
    From the paper:
        "We define the ratio comparing the times at which half of the population 
        is reached in the backbone and the original network, ξ^B_{1/2}, as:

                        ξ^B_{1/2} = t^B_{1/2}  /  t_{1/2}

        In absence of stochastic fluctuations, the aforementioned ratio 
        fulfills ξ^B_{1/2} ⩾ 1, as the metric backbone always removes 
        potential transmission pathways for the virus existing in the original 
        network. In terms of performance, the closer this ratio gets to 
        ξ^B_{1/2} = 1, the more faithful the information provided by the metric 
        backbone is about the dynamics in the entire network."
    """
    
    B = semi_metric_backbone(G)
    t_half_B = t_half(B, sim_time, num_simulations=num_simulations)
    t_half_G = t_half(G, sim_time, num_simulations=num_simulations)
    
    return t_half_B/t_half_G
    

def xi_half_chi(G, chi, sim_time, num_simulations=10):
    """
    From the paper:
        "We define the ratio comparing the times at which half of the population 
        is reached in the backbone and the original network, ξ^B_{1/2}, as:

                        ξ^B_{1/2} = t^B_{1/2}  /  t_{1/2}

        In absence of stochastic fluctuations, the aforementioned ratio 
        fulfills ξ^B_{1/2} ⩾ 1, as the metric backbone always removes 
        potential transmission pathways for the virus existing in the original 
        network. In terms of performance, the closer this ratio gets to 
        ξ^B_{1/2} = 1, the more faithful the information provided by the metric 
        backbone is about the dynamics in the entire network."
    """
    
    B, n = SMDS_sparsifier(G, chi, output_num_components = True)
    
    G_tilde = copy_graph(G)
    until_n_components(G_tilde, n)
    
    t_half_B = t_half(B, sim_time, num_simulations=num_simulations, original_graph_size=G.numberOfNodes())
    t_half_G = t_half(G, sim_time, num_simulations=num_simulations, original_graph_size=G.numberOfNodes())
    t_half_G_tilde = t_half(G_tilde, sim_time, num_simulations=num_simulations, original_graph_size=G.numberOfNodes())
    
    #print(f"t_1/2 SDRG / t_1/2 is {t_half_G_tilde/t_half_G}")
    #print(f"t_1/2 semi-metric / t_1/2 is {t_half_B/t_half_G}")
    print("-- vals --")
    print(f"std: {t_half_G}")
    print(f"smds: {t_half_B}")
    print(f"sdrg: {t_half_G_tilde}")
    return (t_half_G, t_half_B, t_half_G_tilde)
 
def generate_xi_half_curve(G, sim_time, num_simulations):
    
    chi_values = np.linspace(0,1, 5)
    
    sdrg_xi_vals = []
    smds_xi_vals = []  
    
    x = []
    
    for chi in chi_values:
        vals = xi_half_chi(G, chi, sim_time, num_simulations=num_simulations)        
        print(vals)
        sdrg_xi_vals.append(vals[2]/vals[0])
        smds_xi_vals.append(vals[1]/vals[0])
        x.append(chi)
        
    print(sdrg_xi_vals)
    print(smds_xi_vals)
    
    plt.plot(x, sdrg_xi_vals)
    plt.plot(x, smds_xi_vals)
    plt.title("xi^x_{1/2} values for x = SDRG and x = SMDS")
    plt.xlabel('chi value')
    plt.ylabel('xi^x_{1/2}')
    plt.show()
    
    
def compare_data(orig_data, mod_data, original_graph_size, relaxation_time):
    #get prop_time_active_orig
    data_arr = [orig_data, mod_data]
    prop_times_active = np.empty(2)
    
    prop_orig = []
    data = orig_data
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        trans_times = data[site]
        total_time_active = 0
        for i in range(1, len(trans_times), 2):
            total_time_active += trans_times[i] - trans_times[i-1]
        total_time_active_list[site] = total_time_active
    
    proportions_of_time_active = total_time_active_list/relaxation_time
    prop_orig = proportions_of_time_active
    
    
    prop_mod = []
    data = mod_data
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        trans_times = data[site]
        total_time_active = 0
        for i in range(1, len(trans_times), 2):
            total_time_active += trans_times[i] - trans_times[i-1]
        total_time_active_list[site] = total_time_active
    
    proportions_of_time_active = total_time_active_list/relaxation_time
    prop_mod = proportions_of_time_active
    
    comp = abs(prop_orig - np.ones(len(prop_orig))*0.5)
    return np.average(comp)


def Wasserstein_ATES(G, G_tilde, sim_time, start_condition='l', num_simulations=10, dispersion_density=0.01, p=False):
    """
    Pass in original graph G, sparse graph G_tilde, time to simulate for, and 
    the start condition ('l' for localized, 'd' for dispersed), the number of 
    simulations, and the dispersal density to be used if the dispersed initial 
    condition is chosen. 
    The program will run num_simulations simulations, 
    """
    u_values = np.empty((G.numberOfNodes(), num_simulations))
    v_values = np.empty((G.numberOfNodes(), num_simulations))
    
    if start_condition == 'l':
        reset_DCP(G)
        reset_DCP(G_tilde)
    else:
        reset_to_distributed_infection(G, dispersion_density)
        reset_to_distributed_infection(G_tilde, dispersion_density)
        
    for i in range(num_simulations):
        
        # First we need to find a valid starting point that is shared between G and G_tilde
        
        found_valid_common_starting_point = False
        cc = nk.components.ConnectedComponents(G_tilde)
        cc.run()
        
        sizes = cc.getComponentSizes()  # Returns a dict: {component_id: size} w/ sizes of all components
        largest_component_id = max(sizes, key=sizes.get)
        num_attempts = 0
        G_node_count = G.numberOfNodes()
        #G_tilde_nodes = [u for u in G_tilde.iterNodes()]
        start_point = -1
        while not found_valid_common_starting_point:
            start_point = np.random.randint(0, high=G_node_count)
            if num_attempts > G_node_count:
                start_point = num_attempts-G_node_count
                
            cluster_node_is_in = get_site_location(start_point, G_tilde)
            #get_site_location returns a -1 if the site doesn't exist (only when it gets site decimated in sdrg)
            if cluster_node_is_in > -1:
                node_component_id = cc.componentOfNode(cluster_node_is_in)
                
                if node_component_id == largest_component_id and G_tilde.degree(cluster_node_is_in) > 0:
                    found_valid_common_starting_point = True
                    
            num_attempts += 1
            if num_attempts > 2*G_node_count: 
                #true iff we've tried n random samples then gone sequentially 
                # through all nodes, so guaranteed that no valid node exists
                start_point = -1
                found_valid_common_starting_point = True
                
        if start_point == -1: 
            # if there is no valid starting point then give a big penalty
            for site in range(G.numberOfNodes()):
                u_values[site, i] = 0
                v_values[site, i] = sim_time
        else:
            data_dense = sparsified_DCP_fast(G, t_max=sim_time, original_graph_size=G.numberOfNodes(), random_start=False, start_node=start_point)
            data_sparse = sparsified_DCP_fast(G_tilde, t_max=sim_time, original_graph_size=G.numberOfNodes(), random_start=False, start_node=start_point)
            if p:
                print("---------------------")
                print(data_dense)
                print(data_sparse)
            for site in data_dense:#could use data_dense or data_sparse here, it gets the same thing
                #print(site)
                #print(data_dense[site])
                
                if data_dense[site] == [] and data_sparse[site] == []:
                    u_values[site, i] = 0
                    v_values[site, i] = 0
                    #print("site wasn't infected in either")
                elif data_dense[site] != [] and data_sparse[site] == []:
                    u_values[site, i] = 0
                    v_values[site, i] = sim_time
                elif data_dense[site] == [] and data_sparse[site] != []:
                    u_values[site, i] = 0
                    v_values[site, i] = sim_time
                else:
                    #print("")
                    u_values[site, i] = data_dense[site][0]
                    v_values[site, i] = data_sparse[site][0]
                
            
        #print(u_values)
        #print("-")
        #print(v_values)
                
        if start_condition == 'l':
            reset_DCP(G)
            reset_DCP(G_tilde)
        else:
            reset_to_distributed_infection(G, dispersion_density)
            reset_to_distributed_infection(G_tilde, dispersion_density)
                
    wasserstein_distances = []
    for i in range(G.numberOfNodes()):
        wd = wasserstein_distance(u_values[i, :], v_values[i, :])
        wasserstein_distances.append(wd)
    
    return np.average(wasserstein_distances)


#all of the SIR ones here don't really work/get stuck a lot because they don't know when to stop usually.

def Wasserstein_ATES_SIR_steps(G, G_tilde, sim_time, start_condition='l', num_simulations=10, dispersion_density=0.01, p=False):
    """
    Pass in original graph G, sparse graph G_tilde, time to simulate for, and 
    the start condition ('l' for localized, 'd' for dispersed), the number of 
    simulations, and the dispersal density to be used if the dispersed initial 
    condition is chosen. 
    The program will run num_simulations simulations, 
    """
    u_values = np.empty((G.numberOfNodes(), num_simulations))
    v_values = np.empty((G.numberOfNodes(), num_simulations))
    
    if start_condition == 'l':
        reset_DCP(G)
        reset_DCP(G_tilde)
    else:
        reset_to_distributed_infection(G, dispersion_density)
        reset_to_distributed_infection(G_tilde, dispersion_density)
        
    for i in range(num_simulations):
        
        # First we need to find a valid starting point that is shared between G and G_tilde
        
        found_valid_common_starting_point = False
        cc = nk.components.ConnectedComponents(G_tilde)
        cc.run()
        
        sizes = cc.getComponentSizes()  # Returns a dict: {component_id: size} w/ sizes of all components
        largest_component_id = max(sizes, key=sizes.get)
        num_attempts = 0
        G_node_count = G.numberOfNodes()
        #G_tilde_nodes = [u for u in G_tilde.iterNodes()]
        start_point = -1
        while not found_valid_common_starting_point:
            start_point = np.random.randint(0, high=G_node_count)
            if num_attempts > G_node_count:
                start_point = num_attempts-G_node_count
                
            cluster_node_is_in = get_site_location(start_point, G_tilde)
            #get_site_location returns a -1 if the site doesn't exist (only when it gets site decimated in sdrg)
            if cluster_node_is_in > -1:
                node_component_id = cc.componentOfNode(cluster_node_is_in)
                
                if node_component_id == largest_component_id and G_tilde.degree(cluster_node_is_in) > 0:
                    found_valid_common_starting_point = True
                    
            num_attempts += 1
            if num_attempts > 2*G_node_count: 
                #true iff we've tried n random samples then gone sequentially 
                # through all nodes, so guaranteed that no valid node exists
                start_point = -1
                found_valid_common_starting_point = True
                
        if start_point == -1: 
            # if there is no valid starting point then give a big penalty
            for site in range(G.numberOfNodes()):
                u_values[site, i] = 0
                v_values[site, i] = sim_time
        else:
            data_dense = sparsified_DCP_fast_SIR_steps(G, steps_max=sim_time, original_graph_size=G.numberOfNodes(), random_start=False, start_node=start_point)
            data_sparse = sparsified_DCP_fast_SIR_steps(G_tilde, steps_max=sim_time, original_graph_size=G.numberOfNodes(), random_start=False, start_node=start_point)
            if p:
                print("---------------------")
                print(data_dense)
                print(data_sparse)
            for site in data_dense:#could use data_dense or data_sparse here, it gets the same thing
                #print(site)
                #print(data_dense[site])
                
                if data_dense[site] == [] and data_sparse[site] == []:
                    u_values[site, i] = 0
                    v_values[site, i] = 0
                    #print("site wasn't infected in either")
                elif data_dense[site] != [] and data_sparse[site] == []:
                    u_values[site, i] = 0
                    v_values[site, i] = sim_time
                elif data_dense[site] == [] and data_sparse[site] != []:
                    u_values[site, i] = 0
                    v_values[site, i] = sim_time
                else:
                    #print("")
                    u_values[site, i] = data_dense[site][0]
                    v_values[site, i] = data_sparse[site][0]
                
            
        #print(u_values)
        #print("-")
        #print(v_values)
                
        if start_condition == 'l':
            reset_DCP(G)
            reset_DCP(G_tilde)
        else:
            reset_to_distributed_infection(G, dispersion_density)
            reset_to_distributed_infection(G_tilde, dispersion_density)
                
    wasserstein_distances = []
    for i in range(G.numberOfNodes()):
        wd = wasserstein_distance(u_values[i, :], v_values[i, :])
        wasserstein_distances.append(wd)
    
    return np.average(wasserstein_distances)


def Wasserstein_ATES_SIR(G, G_tilde, sim_time, start_condition='l', num_simulations=10, dispersion_density=0.01, p=False):
    """
    Pass in original graph G, sparse graph G_tilde, time to simulate for, and 
    the start condition ('l' for localized, 'd' for dispersed), the number of 
    simulations, and the dispersal density to be used if the dispersed initial 
    condition is chosen. 
    The program will run num_simulations simulations, 
    """
    u_values = np.empty((G.numberOfNodes(), num_simulations))
    v_values = np.empty((G.numberOfNodes(), num_simulations))
    
    if start_condition == 'l':
        reset_DCP(G)
        reset_DCP(G_tilde)
    else:
        reset_to_distributed_infection(G, dispersion_density)
        reset_to_distributed_infection(G_tilde, dispersion_density)
        
    for i in range(num_simulations):
        
        # First we need to find a valid starting point that is shared between G and G_tilde
        
        found_valid_common_starting_point = False
        cc = nk.components.ConnectedComponents(G_tilde)
        cc.run()
        
        sizes = cc.getComponentSizes()  # Returns a dict: {component_id: size} w/ sizes of all components
        largest_component_id = max(sizes, key=sizes.get)
        num_attempts = 0
        G_node_count = G.numberOfNodes()
        #G_tilde_nodes = [u for u in G_tilde.iterNodes()]
        start_point = -1
        while not found_valid_common_starting_point:
            start_point = np.random.randint(0, high=G_node_count)
            if num_attempts > G_node_count:
                start_point = num_attempts-G_node_count
                
            cluster_node_is_in = get_site_location(start_point, G_tilde)
            #get_site_location returns a -1 if the site doesn't exist (only when it gets site decimated in sdrg)
            if cluster_node_is_in > -1:
                node_component_id = cc.componentOfNode(cluster_node_is_in)
                
                if node_component_id == largest_component_id and G_tilde.degree(cluster_node_is_in) > 0:
                    found_valid_common_starting_point = True
                    
            num_attempts += 1
            if num_attempts > 2*G_node_count: 
                #true iff we've tried n random samples then gone sequentially 
                # through all nodes, so guaranteed that no valid node exists
                start_point = -1
                found_valid_common_starting_point = True
                
        if start_point == -1: 
            # if there is no valid starting point then give a big penalty
            for site in range(G.numberOfNodes()):
                u_values[site, i] = 0
                v_values[site, i] = sim_time
        else:
            data_dense = sparsified_DCP_fast_SIR(G, t_max=sim_time, original_graph_size=G.numberOfNodes(), random_start=False, start_node=start_point)
            data_sparse = sparsified_DCP_fast_SIR(G_tilde, t_max=sim_time, original_graph_size=G.numberOfNodes(), random_start=False, start_node=start_point)
            if p:
                print("---------------------")
                print(data_dense)
                print(data_sparse)
            for site in data_dense:#could use data_dense or data_sparse here, it gets the same thing
                #print(site)
                #print(data_dense[site])
                
                if data_dense[site] == [] and data_sparse[site] == []:
                    u_values[site, i] = 0
                    v_values[site, i] = 0
                    #print("site wasn't infected in either")
                elif data_dense[site] != [] and data_sparse[site] == []:
                    u_values[site, i] = 0
                    v_values[site, i] = sim_time
                elif data_dense[site] == [] and data_sparse[site] != []:
                    u_values[site, i] = 0
                    v_values[site, i] = sim_time
                else:
                    #print("")
                    u_values[site, i] = data_dense[site][0]
                    v_values[site, i] = data_sparse[site][0]
                
            
        #print(u_values)
        #print("-")
        #print(v_values)
                
        if start_condition == 'l':
            reset_DCP(G)
            reset_DCP(G_tilde)
        else:
            reset_to_distributed_infection(G, dispersion_density)
            reset_to_distributed_infection(G_tilde, dispersion_density)
                
    wasserstein_distances = []
    for i in range(G.numberOfNodes()):
        wd = wasserstein_distance(u_values[i, :], v_values[i, :])
        wasserstein_distances.append(wd)
    
    return np.average(wasserstein_distances)



def generate_ATES_comparison_curve(G, sim_time, start_condition='l', num_simulations=10, dispersion_density=0.01):
    k = 5 #the number of increments of sparsification
    initial_num_components = number_of_components(G)
    
    #store the arrival time estimation scores for both methosd in these lists
    sdrg_ATES = []
    effR_ATES = []
    x = []
    """
        a = G_sdrg.numberOfNodes()
        
        print(a)
        G_sdrg_size = a + G_sdrg.numberOfEdges()
        
        
        #print(G)
        #print(G_sdrg)
        while G_sdrg_size > target_num_components:
            G_sdrg = sdrg_step(G_sdrg)
            a -= 1
            G_sdrg_size = a + G_sdrg.numberOfEdges()
    """
        
    
    for i in range(1, k+1):
        
        
        
        #each sparsification increment, find the effR and sdrg sparsifications
        target_num_components = (i/k)*G.numberOfEdges() + G.numberOfNodes()
        
        x.append(target_num_components/initial_num_components)
    
        print(f"----- RUNNING {100*target_num_components/initial_num_components}% SPARSIFICATION SIMULATIONS -----")
        G_sdrg = copy_graph(G)
        G_effR = copy_graph(G)
        
        print("sparsifying G")
        decimated_sites=[]
        while number_of_components(G_sdrg) > target_num_components:
            G_sdrg, decimated_sites = sdrg_step(G_sdrg, decimated_sites, visualizeStep=False)
            
            
        #G_sdrg = prop_sdrg(G_sdrg, i/k)
        
        
        target_effR_edgecount = int(((k-i)/k)*G.numberOfEdges())
        G_effR = effective_resistance_sampling_sparsification(G_effR, target_effR_edgecount)
        print("")
        print("- running sdrg ATES test -")
        w_ates_sdrg = Wasserstein_ATES(G, G_sdrg, sim_time, num_simulations=num_simulations)
        print(f"ATES FOR SDRG = {w_ates_sdrg}")
        print("")
        print("- running effR ATES test -")
        w_ates_effR = Wasserstein_ATES(G, G_effR, sim_time, num_simulations=num_simulations)
        print(f"ATES FOR effR = {w_ates_effR}")
        
        sdrg_ATES.append(w_ates_sdrg)
        effR_ATES.append(w_ates_effR)
    
    print(sdrg_ATES)
    print(effR_ATES)
    
    #x = np.arange(0.0, k)
    #sdrg_ATES = [np.float64(26.568118327452215), np.float64(13.337340450566153), np.float64(8.380268554475983), np.float64(6.200337406177095), np.float64(5.1174552012518415), np.float64(5.741873695984299), np.float64(4.3541790614438), np.float64(4.049318993134221), np.float64(3.372839426655192)]
    #effR_ATES = [np.float64(1.702880859375), np.float64(1.982421875), np.float64(2.0434570312499996), np.float64(2.177734375), np.float64(2.3571777343749996), np.float64(1.8473770965020069), np.float64(1.9188615108874316), np.float64(1.8261718749999996), np.float64(2.108154296875)]
    
    plt.plot(x, sdrg_ATES)
    plt.plot(x, effR_ATES)
    plt.title("A basic plot using pyplot")
    plt.xlabel('sparsification percentage')
    plt.ylabel('Arrival Time Error Score')
    plt.show()


def generate_ATES_comparison_curve_range(G, sim_time, start_condition='l', num_simulations=10, dispersion_density=0.01, sparsification_range=[0.333333333,0.34]):
    k = 8 #the number of increments of sparsification
    initial_num_components = number_of_components(G)
    
    #store the arrival time estimation scores for both methosd in these lists
    sdrg_ATES = []
    effR_ATES = []
    x = []
    """
        a = G_sdrg.numberOfNodes()
        
        print(a)
        G_sdrg_size = a + G_sdrg.numberOfEdges()
        
        
        #print(G)
        #print(G_sdrg)
        while G_sdrg_size > target_num_components:
            G_sdrg = sdrg_step(G_sdrg)
            a -= 1
            G_sdrg_size = a + G_sdrg.numberOfEdges()
    """
        
    sparsification_proportions = np.linspace(sparsification_range[0], sparsification_range[1], k)
    
    print(initial_num_components*sparsification_proportions)
    print((initial_num_components*sparsification_proportions)-4096)
    
    for proportion in sparsification_proportions:
        
        
        
        #each sparsification increment, find the effR and sdrg sparsifications
        target_num_components = proportion*number_of_components(G)
        
        x.append(target_num_components/initial_num_components)
    
        print(f"----- RUNNING {100*target_num_components/initial_num_components}% SPARSIFICATION SIMULATIONS -----")
        G_sdrg = copy_graph(G)
        G_effR = copy_graph(G)
        
        print("sparsifying G")
        decimated_sites=[]
        while number_of_components(G_sdrg) > target_num_components:
            G_sdrg, decimated_sites = sdrg_step(G_sdrg, decimated_sites, visualizeStep=False)
            
        print(f"G_sdrg has {number_of_components(G_sdrg)}, target was {target_num_components}")
            
        #G_sdrg = prop_sdrg(G_sdrg, i/k)
        
        
        target_effR_edgecount = max(int(target_num_components-G.numberOfNodes()), 0)
        
        G_effR = effective_resistance_sampling_sparsification(G_effR, target_effR_edgecount)
        print("")
        print("- running sdrg ATES test -")
        w_ates_sdrg = Wasserstein_ATES(G, G_sdrg, sim_time, num_simulations=num_simulations)
        print(f"ATES FOR SDRG = {w_ates_sdrg}")
        print("")
        print("- running effR ATES test -")
        if target_effR_edgecount > 0:
            w_ates_effR = Wasserstein_ATES(G, G_effR, sim_time, num_simulations=num_simulations, p=True)
            print(f"ATES FOR effR = {w_ates_effR}")
        else:
            w_ates_effR = sim_time
            print(f"ATES FOR effR = {w_ates_effR}")
        sdrg_ATES.append(w_ates_sdrg)
        effR_ATES.append(w_ates_effR)
    
    print(sdrg_ATES)
    print(effR_ATES)
    
    #x = np.arange(0.0, k)
    #sdrg_ATES = [np.float64(26.568118327452215), np.float64(13.337340450566153), np.float64(8.380268554475983), np.float64(6.200337406177095), np.float64(5.1174552012518415), np.float64(5.741873695984299), np.float64(4.3541790614438), np.float64(4.049318993134221), np.float64(3.372839426655192)]
    #effR_ATES = [np.float64(1.702880859375), np.float64(1.982421875), np.float64(2.0434570312499996), np.float64(2.177734375), np.float64(2.3571777343749996), np.float64(1.8473770965020069), np.float64(1.9188615108874316), np.float64(1.8261718749999996), np.float64(2.108154296875)]
    
    plt.plot(x, sdrg_ATES)
    plt.plot(x, effR_ATES)
    plt.title("Arrival Time Error Scores of SDRG v.s. effR Sparsification")
    plt.xlabel('sparsification percentage')
    plt.ylabel('Arrival Time Error Score')
    plt.show()


def generate_ATES_comparison_curve_range_SIR_steps(G, sim_time, start_condition='l', num_simulations=10, dispersion_density=0.01, sparsification_range=[0.333333333,0.34]):
    k = 8 #the number of increments of sparsification
    initial_num_components = number_of_components(G)
    
    #store the arrival time estimation scores for both methosd in these lists
    sdrg_ATES = []
    effR_ATES = []
    x = []
    """
        a = G_sdrg.numberOfNodes()
        
        print(a)
        G_sdrg_size = a + G_sdrg.numberOfEdges()
        
        
        #print(G)
        #print(G_sdrg)
        while G_sdrg_size > target_num_components:
            G_sdrg = sdrg_step(G_sdrg)
            a -= 1
            G_sdrg_size = a + G_sdrg.numberOfEdges()
    """
        
    sparsification_proportions = np.linspace(sparsification_range[0], sparsification_range[1], k)
    
    print(initial_num_components*sparsification_proportions)
    print((initial_num_components*sparsification_proportions)-4096)
    
    for proportion in sparsification_proportions:
        
        
        
        #each sparsification increment, find the effR and sdrg sparsifications
        target_num_components = proportion*number_of_components(G)
        
        x.append(target_num_components/initial_num_components)
    
        print(f"----- RUNNING {100*target_num_components/initial_num_components}% SPARSIFICATION SIMULATIONS -----")
        G_sdrg = copy_graph(G)
        G_effR = copy_graph(G)
        
        print("sparsifying G")
        decimated_sites=[]
        while number_of_components(G_sdrg) > target_num_components:
            G_sdrg, decimated_sites = sdrg_step(G_sdrg, decimated_sites, visualizeStep=False)
            
        print(f"G_sdrg has {number_of_components(G_sdrg)}, target was {target_num_components}")
            
        #G_sdrg = prop_sdrg(G_sdrg, i/k)
        
        
        target_effR_edgecount = max(int(target_num_components-G.numberOfNodes()), 0)
        
        G_effR = effective_resistance_sampling_sparsification(G_effR, target_effR_edgecount)
        print("")
        print("- running sdrg ATES test -")
        w_ates_sdrg = Wasserstein_ATES_SIR_steps(G, G_sdrg, sim_time, num_simulations=num_simulations)
        print(f"ATES FOR SDRG = {w_ates_sdrg}")
        print("")
        print("- running effR ATES test -")
        if target_effR_edgecount > 0:
            w_ates_effR = Wasserstein_ATES_SIR_steps(G, G_effR, sim_time, num_simulations=num_simulations, p=True)
            print(f"ATES FOR effR = {w_ates_effR}")
        else:
            w_ates_effR = sim_time
            print(f"ATES FOR effR = {w_ates_effR}")
        sdrg_ATES.append(w_ates_sdrg)
        effR_ATES.append(w_ates_effR)
    
    print(sdrg_ATES)
    print(effR_ATES)
    
    #x = np.arange(0.0, k)
    #sdrg_ATES = [np.float64(26.568118327452215), np.float64(13.337340450566153), np.float64(8.380268554475983), np.float64(6.200337406177095), np.float64(5.1174552012518415), np.float64(5.741873695984299), np.float64(4.3541790614438), np.float64(4.049318993134221), np.float64(3.372839426655192)]
    #effR_ATES = [np.float64(1.702880859375), np.float64(1.982421875), np.float64(2.0434570312499996), np.float64(2.177734375), np.float64(2.3571777343749996), np.float64(1.8473770965020069), np.float64(1.9188615108874316), np.float64(1.8261718749999996), np.float64(2.108154296875)]
    
    plt.plot(x, sdrg_ATES)
    plt.plot(x, effR_ATES)
    plt.title("Arrival Time Error Scores of SDRG v.s. effR Sparsification")
    plt.xlabel('sparsification percentage')
    plt.ylabel('Arrival Time Error Score')
    plt.show()


def generate_ATES_comparison_curve_range_SIR_time(G, sim_time, start_condition='l', num_simulations=10, dispersion_density=0.01, sparsification_range=[0.333333333,0.34]):
    k = 8 #the number of increments of sparsification
    initial_num_components = number_of_components(G)
    
    #store the arrival time estimation scores for both methosd in these lists
    sdrg_ATES = []
    effR_ATES = []
    x = []
    """
        a = G_sdrg.numberOfNodes()
        
        print(a)
        G_sdrg_size = a + G_sdrg.numberOfEdges()
        
        
        #print(G)
        #print(G_sdrg)
        while G_sdrg_size > target_num_components:
            G_sdrg = sdrg_step(G_sdrg)
            a -= 1
            G_sdrg_size = a + G_sdrg.numberOfEdges()
    """
        
    sparsification_proportions = np.linspace(sparsification_range[0], sparsification_range[1], k)
    
    print(initial_num_components*sparsification_proportions)
    print((initial_num_components*sparsification_proportions)-4096)
    
    for proportion in sparsification_proportions:
        
        
        
        #each sparsification increment, find the effR and sdrg sparsifications
        target_num_components = proportion*number_of_components(G)
        
        x.append(target_num_components/initial_num_components)
    
        print(f"----- RUNNING {100*target_num_components/initial_num_components}% SPARSIFICATION SIMULATIONS -----")
        G_sdrg = copy_graph(G)
        G_effR = copy_graph(G)
        
        print("sparsifying G")
        decimated_sites=[]
        while number_of_components(G_sdrg) > target_num_components:
            G_sdrg, decimated_sites = sdrg_step(G_sdrg, decimated_sites, visualizeStep=False)
            
        print(f"G_sdrg has {number_of_components(G_sdrg)}, target was {target_num_components}")
            
        #G_sdrg = prop_sdrg(G_sdrg, i/k)
        
        
        target_effR_edgecount = max(int(target_num_components-G.numberOfNodes()), 0)
        
        G_effR = effective_resistance_sampling_sparsification(G_effR, target_effR_edgecount)
        print("")
        print("- running sdrg ATES test -")
        w_ates_sdrg = Wasserstein_ATES_SIR(G, G_sdrg, sim_time, num_simulations=num_simulations)
        print(f"ATES FOR SDRG = {w_ates_sdrg}")
        print("")
        print("- running effR ATES test -")
        if target_effR_edgecount > 0:
            w_ates_effR = Wasserstein_ATES_SIR(G, G_effR, sim_time, num_simulations=num_simulations)
            print(f"ATES FOR effR = {w_ates_effR}")
        else:
            w_ates_effR = sim_time
            print(f"ATES FOR effR = {w_ates_effR}")
        sdrg_ATES.append(w_ates_sdrg)
        effR_ATES.append(w_ates_effR)
    
    print(sdrg_ATES)
    print(effR_ATES)
    
    #x = np.arange(0.0, k)
    #sdrg_ATES = [np.float64(26.568118327452215), np.float64(13.337340450566153), np.float64(8.380268554475983), np.float64(6.200337406177095), np.float64(5.1174552012518415), np.float64(5.741873695984299), np.float64(4.3541790614438), np.float64(4.049318993134221), np.float64(3.372839426655192)]
    #effR_ATES = [np.float64(1.702880859375), np.float64(1.982421875), np.float64(2.0434570312499996), np.float64(2.177734375), np.float64(2.3571777343749996), np.float64(1.8473770965020069), np.float64(1.9188615108874316), np.float64(1.8261718749999996), np.float64(2.108154296875)]
    
    plt.plot(x, sdrg_ATES)
    plt.plot(x, effR_ATES)
    plt.title("Arrival Time Error Scores of SDRG v.s. effR Sparsification")
    plt.xlabel('sparsification percentage')
    plt.ylabel('Arrival Time Error Score')
    plt.show()


def WDMeans(Org_Arrivals, Sps_Arrivals, simnum):
    arrivals = []
    for i in range(len(Org_Arrivals)):
        arrival_org = [x / simnum for x in Org_Arrivals[i]]
        arrival_spl = [x / simnum for x in Sps_Arrivals[i]]
        if arrival_org == [] or arrival_spl == []:
            if arrival_org == [] and arrival_spl != []:
                arrivals.append(1)
            elif arrival_org != [] and arrival_spl == []:
                arrivals.append(1)
            elif arrival_org == [] and arrival_spl == []:
                arrivals.append(0)
        else:
            arrivals.append(wasserstein_distance(arrival_org, arrival_spl))

    return np.mean(arrivals)


""" ------------------------------------ UTILITIES ------------------------------------ """

def get_site_location(site, G):
    components = G.getNodeAttribute("components", str)
    node_list = [u for u in G.iterNodes()]
    #print(start_node in node_list)
    if site in node_list:
        return site
    else:
        for node in G.iterNodes():
            c = components[node].split("_")
            #print(f"site {site} components are {c}")
            if str(site) in c:
                #print(site)
                return node
    return -1

def site_exists_in_clusters(site, G):
    components = G.getNodeAttribute("components", str)
    node_list = [u for u in G.iterNodes()]
    #print(start_node in node_list)
    if site in node_list:
        return True
    else:
        for node in G.iterNodes():
            c = components[node].split("_")
            #print(f"site {site} components are {c}")
            if str(site) in c:
                #print(site)
                return True
    return False

def number_of_components(G):
    return G.numberOfNodes() + G.numberOfEdges()

def complexity_score(G):
    score = 0
    for u in G.iterNodes():
        score += G.degree(u)
    return score

def copy_graph(G_in):
    G = nk.graph.Graph(n=G_in.numberOfNodes(), weighted=True, edgesIndexed=True)
    
    in_activity = G_in.getNodeAttribute("active", int)
    in_mu = G_in.getNodeAttribute("mu", float)
    in_comp = G_in.getNodeAttribute("components", str)
    for edge in G_in.iterEdges():
        G.addEdge(edge[0], edge[1], w=G_in.weight(edge[0], edge[1]))
    
    is_active = G.attachNodeAttribute("active", int)
    healing_factor = G.attachNodeAttribute("mu", float)
    components = G.attachNodeAttribute("components", str)
    #add disorder to healing values too if desired
    
    for u in G.iterNodes():
        healing_factor[u] = in_mu[u]
        is_active[u] = in_activity[u]
        components[u] = in_comp[u]
    
    return G


def print_graph_values(G):
    healing_factor = G.getNodeAttribute("mu", float)
    num_edges = G.numberOfEdges()
    num_nodes = G.numberOfNodes()
    print(f"G has {num_edges} edges and {num_nodes} nodes \n------------------")
    
    for u in G.iterNodes():
        print(f"node {u} has mu = {healing_factor[u]}")
    for u,v in G.iterEdges():
        print(f"edge from {u} to {v} has lambda = {G.weight(u,v)}")
    
    
    if num_edges > 0:
        lambda_avg = (G.totalEdgeWeight()/num_edges)
        print(f"The average lambda = {lambda_avg}")
        
        
def visualize(G, active_nodes=[], recovered_nodes=[], node_size = 100):
    #G = nk.readGraph("../input/karate.graph", nk.Format.METIS)
    # Initalize and run Betweenness algorithm
    
    nx_graph = nk.nxadapter.nk2nx(G)

    # 3. Draw the graph using Matplotlib
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(nx_graph, seed=2)

    
    options = {"edgecolors": "tab:gray", "node_size": node_size*1.2, "alpha": 0.9}
    nx.draw(nx_graph, pos=pos, with_labels=True, node_color='skyblue', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=active_nodes, node_color="tab:red", **options)
    plt.show()
    

def vis_lat(data, dimensions):
    matrix = np.matrix(data.to_numpy())
    
    #print(matrix.shape)

    proportions_of_time_active_listform = matrix.sum(axis=1)
    proportions_of_time_active = [u[0,0] for u in proportions_of_time_active_listform]
    
    #print(proportions_of_time_active)
    
    h = dimensions[0]
    w = dimensions[1]
    
    Z = np.zeros((h,w))
    
    #first build a bunch of lines/rows
    for y in range(h):
        for x in range(w):  
            Z[x,y] = proportions_of_time_active[x+(y*w)]
    
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, Z, cmap='viridis')
    fig.tight_layout()

    #fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
    
    # plot just the positive data and save the
    # color "mappable" object returned by ax1.imshow
    pos = ax.imshow(Z, cmap='viridis', interpolation='none')
    
    # add the colorbar using the figure's method,
    # telling which mappable we're talking about and
    # which Axes object it should be near
    fig.colorbar(pos, ax=ax)    
    plt.show()
    
    
def get_proportions(simulation_data):
    proportions_arr = []
    for df in simulation_data:
        df_proportions = []
        for col in df:
            num_active = sum(col)
            print(num_active)
            df_proportions.append(num_active/df.shape[0])
            
        proportions_arr.append(df_proportions)
    
    
    # Concatenate along a new axis and take the mean
    df_average = pd.concat(simulation_data).groupby(level=0).mean()
    print(df_average)
    """
    We want to calculate the time of arrival, represented by list with the 
    radius of infection spread at some standardized timesteps; maybe like every 
    0.1 seconds or similar. Radius is determined using bond lengths, where each
    bond length l_ij = 1/lambda_ij.
    
    This function will simply take in a graph, then run a number of DCP 
    simulations on it (n_sims giving the number) and average the arrays of 
    radius/time for each one.
    
    These simulations will have many non-surviving runs, so we will make the program more efficient by 
    
    """


def swap_column_vals_after_k(df, row, k):
    for i in range(k, df.shape[1]):
        if df.iat[row, i] == 0:
            df.iat[row, i] = 1
        else:
            df.iat[row, i] = 0
    return df

def repeated_sparsifications_simulations_and_visualizations():
    G = generate_square_lattice(100, 100)
    #reset_DCP(G)
    all_data = []
    for i in range(6):
        reset_to_distributed_infection(G, 0.7)
        data = fast_sparsified_asymptotic_quasistationary_activity_probability(G, G_structure='lattice', original_graph_size=10000, dimensions=[100,100])
        all_data.append(data)
        G = prop_sdrg(G, 0.5)
        
    return all_data

def fast_random_choose(s, min_val, max_val):
    for i in range(100):
        u = np.random.randint(min_val, high=max_val)
        if u in s:
            return u
    return np.random.choice(list(s))
        
def test():
    
    data = set(np.random.randint(0, high=10000, size=12000))
    print(f"size is {len(data)}")
    sum_time0 = 0
    sum_time1 = 0
    
    for i in range(1000):
        t0 = time.time_ns()
        a = fast_random_choose(data, 0, 10000)
        t1 = time.time_ns()
        sum_time0 += (t1-t0)/1000000000
        
    for i in range(1000):
        t0 = time.time_ns()
        np.random.choice(list(data))
        t1 = time.time_ns()
        sum_time1 += (t1-t0)/1000000000
    
    print(f"our func has t_avg = {sum_time0/1000}")
    print(f"their func has t_avg = {sum_time1/1000}")
"""
look at pearson or spearman correlation coeff
least squares? would have to normalize data then
kl-divergrence? 2D wasserstein is probably better!
check xi_1/2 with SMDS against effR 
run sdrg until some omega!
change how sdrg networks are being generated;
- check using maximum rule which clusters and edges get removed from maximum rule 
- write in logbook 



- Check weight based, effR, semi-metric, etc
- Use quasistationary simulation (don't let infection ever die out)
- how did the clusters get determined for DCP from the 2020 paper?
    - which metrics are being tracked in the 2020 paper from the DCP there?
"""

