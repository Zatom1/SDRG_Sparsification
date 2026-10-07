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
from scipy.optimize import curve_fit
import bisect
import pickle
import matplotlib.colors as mcolors
from numba import jit
from math import log as ln
import seaborn as sns
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
        
        
def generate_square_lattice(width, height, torus = True, visualize_on=False, disordered_mu = True, constant_mu = 1.0, disordered_lambda=True, constant_lambda = 0.8, chain=True, disorder_alpha = 4, critical=True):
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
                
    
    if chain:
        G.removeSelfLoops()
    
    if visualize_on:
        print("generated; now visualizing")
        visualize(G)
    
    
    """if disordered_lambda:
        nk.graphtools.randomizeWeights(G) #these weights act as lambda vals
    else:
        for u,v in G.iterEdges():
            G.setWeight(u,v,constant_lambda)"""
    
    #init activity values (infected/healed)
    is_active = G.attachNodeAttribute("active", int)
    
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    
    #node & edge components have the *like* components that have merged to form any given edge
    components = G.attachNodeAttribute("components", str)  #like "12_53_25_65"
    edge_components = G.attachEdgeAttribute("e_comp", str) #like "(12,53)_(25,65)"
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = G.attachNodeAttribute("mu_comp", str)
    lambda_components = G.attachEdgeAttribute("lambda_comp", str)
    infection_progenitor = G.attachNodeAttribute("inf_prog", int)
    for site in G.iterNodes():
        infection_progenitor[site] = site
    
    #add disorder to healing values too if desired
    if critical:
        N = G.numberOfNodes()
        #mu_values = generate_power_law_log_mean(N, disorder_alpha, 1.6784)
        #print(mu_values)
        #lambda_values = generate_power_law_log_mean(G.numberOfEdges(), 1, 1)
        for u in G.iterNodes():
            #mu = scipy.stats.powerlaw.rvs(disorder_alpha, size=1)
            #print(mu)
            mu = np.random.random()*math.exp(1.6784)
            healing_factor[u] = mu#mu_values[u]#[0]
            is_active[u] = 0
            components[u] = f"{u}"
            mu_components[u] = f"n{u}"
        for edge in G.iterEdges():
            lambda_val = np.random.random()
            
            G.setWeight(edge[0], edge[1], lambda_val)
            eid = G.edgeId(edge[0], edge[1])
            edge_components[eid] = f"{edge}"
            lambda_components[eid] = f"e({edge[0]},{edge[1]})"
    else:
        if disordered_mu:
            for u in G.iterNodes():
                #mu = scipy.stats.powerlaw.rvs(disorder_alpha, size=1)
                #print(mu)
                mu = np.random.random()
                healing_factor[u] = mu#[0]
                is_active[u] = 0
                components[u] = f"{u}"
                mu_components[u] = f"n{u}"
        else:
            for u in G.iterNodes():
                healing_factor[u] = constant_mu
                is_active[u] = 0
                components[u] = f"{u}"
                mu_components[u] = f"n{u}"
                
        for edge in G.iterEdges():
            lambda_val = scipy.stats.powerlaw.rvs(disorder_alpha, size=1)
            
            G.setWeight(edge[0], edge[1], lambda_val[0])
            eid = G.edgeId(edge[0], edge[1])
            edge_components[eid] = f"{edge}"
            lambda_components[eid] = f"e({edge[0]},{edge[1]})"
    print("done!")
        
    return G

def generate_random_graph(N, avg_degree, degree_stdev, disorder_alpha = 4, viz=False):
    
    G = nk.graph.Graph(n=N, weighted = True, edgesIndexed=True)
    
    #init activity values (infected/healed)
    is_active = G.attachNodeAttribute("active", int)
    infection_progenitor = G.attachNodeAttribute("inf_prog", int)
    for site in G.iterNodes():
        infection_progenitor[site] = site
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    
    #node & edge components have the *like* components that have merged to form any given edge
    components = G.attachNodeAttribute("components", str)  #like "12_53_25_65"
    edge_components = G.attachEdgeAttribute("e_comp", str) #like "(12,53)_(25,65)"
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = G.attachNodeAttribute("mu_comp", str)
    lambda_components = G.attachEdgeAttribute("lambda_comp", str)
    
    for site in G.iterNodes():
        num_edges = max(scipy.stats.norm.rvs(size=1, loc=avg_degree, scale=degree_stdev)[0], 1)
        set_of_other_sites = set([node for node in G.iterNodes()]) - {site}
        
        healing_factor[site] = np.random.random()*avg_degree
        is_active[site] = 0
        components[site] = f"{site}"
        mu_components[site] = f"n{site}"
        
        for edge in range(round(num_edges)):
            
            #it's possible to choose a number greater than the total number of sites; 
            # this prevents that from breaking things
            if len(set_of_other_sites) > 0:
                lambda_val = scipy.stats.powerlaw.rvs(disorder_alpha, size=1)[0]
                
                other_site = np.random.choice(tuple(set_of_other_sites))
                #print(f"{site} - {other_site}")
                G.addEdge(site, other_site, lambda_val)
                
                set_of_other_sites = set_of_other_sites - {other_site}
    
    G.removeMultiEdges()
    for edge in G.iterEdges():
        #edge = (site, other_site)
        eid = G.edgeId(edge[0], edge[1])
        edge_components[eid] = f"{edge}"
        lambda_components[eid] = f"e({edge[0]},{edge[1]})"
    
    
    
    
    
    G.indexEdges()
        
    if viz:
        visualize(G)
    
    return G


def generate_erdos_renyi(N, E, disorder_alpha = 4, viz=False):
    
    G = nk.graph.Graph(n=N, weighted = True, edgesIndexed=True)
    
    #init activity values (infected/healed)
    is_active = G.attachNodeAttribute("active", int)
    infection_progenitor = G.attachNodeAttribute("inf_prog", int)
    for site in G.iterNodes():
        infection_progenitor[site] = site
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    
    #node & edge components have the *like* components that have merged to form any given edge
    components = G.attachNodeAttribute("components", str)  #like "12_53_25_65"
    edge_components = G.attachEdgeAttribute("e_comp", str) #like "(12,53)_(25,65)"
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = G.attachNodeAttribute("mu_comp", str)
    lambda_components = G.attachEdgeAttribute("lambda_comp", str)
    
    if E > (N*(N-1))/2:
        print("given number of edges is too high! returning set of vertices")
        return G
    
    for i in range(E):
        u=0
        v=0
        num_tries = 0
        while u == v or G.weight(u, v) > 0:
            #rechoose v until we get a valid edge that doesn't yet exist
            u = np.random.randint(0, N)
            v = np.random.randint(0, N)
            num_tries+=1
            if num_tries > E:
                print("could not find a valid new edge to add. returning partial graph")
                return G
        lambda_val = scipy.stats.powerlaw.rvs(disorder_alpha, size=1)[0]
        
        G.addEdge(u, v, lambda_val)
    
    for site in G.iterNodes():
        #using the factor of E/N brings us closer to critical point
        healing_factor[site] = np.random.random()*(E/N)
        is_active[site] = 0
        components[site] = f"{site}"
        mu_components[site] = f"n{site}"
        

    #G.removeMultiEdges()
    for edge in G.iterEdges():
        #edge = (site, other_site)
        eid = G.edgeId(edge[0], edge[1])
        edge_components[eid] = f"{edge}"
        lambda_components[eid] = f"e({edge[0]},{edge[1]})"
    
    
    
    
    
    G.indexEdges()
        
    if viz:
        visualize(G)
    
    return G
       

def generate_BA(N, E, viz=False):
    k = E/(N-1)
    print(k)
    bag = nk.generators.BarabasiAlbertGenerator(k, N)
    
    # Run algorithm
    G = bag.generate()
    G.indexEdges()
    G = nk.graphtools.toWeighted(G)
    #init activity values (infected/healed)
    is_active = G.attachNodeAttribute("active", int)
    infection_progenitor = G.attachNodeAttribute("inf_prog", int)
    for site in G.iterNodes():
        infection_progenitor[site] = site
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    
    #node & edge components have the *like* components that have merged to form any given edge
    components = G.attachNodeAttribute("components", str)  #like "12_53_25_65"
    edge_components = G.attachEdgeAttribute("e_comp", str) #like "(12,53)_(25,65)"
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = G.attachNodeAttribute("mu_comp", str)
    lambda_components = G.attachEdgeAttribute("lambda_comp", str)
    
    
    for site in G.iterNodes():
        #using the factor of E/N brings us closer to critical point
        healing_factor[site] = np.random.random()*(E/N)
        is_active[site] = 0
        components[site] = f"{site}"
        mu_components[site] = f"n{site}"
    
    for edge in G.iterEdges():
        #edge = (site, other_site)
        eid = G.edgeId(edge[0], edge[1])
        edge_components[eid] = f"{edge}"
        lambda_components[eid] = f"e({edge[0]},{edge[1]})"
    G.removeSelfLoops()
    if viz:
        visualize(G)
    
    return G
    


def generate_watts_strogatz(N, E, beta, disorder_alpha = 4, viz=False):
    """
    generates a watts-strogatz random graph

    """
    
    
    
    G = nk.graph.Graph(n=N, weighted = True, edgesIndexed=True)
    
    #init activity values (infected/healed)
    is_active = G.attachNodeAttribute("active", int)
    infection_progenitor = G.attachNodeAttribute("inf_prog", int)
    for site in G.iterNodes():
        infection_progenitor[site] = site
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    
    #node & edge components have the *like* components that have merged to form any given edge
    components = G.attachNodeAttribute("components", str)  #like "12_53_25_65"
    edge_components = G.attachEdgeAttribute("e_comp", str) #like "(12,53)_(25,65)"
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = G.attachNodeAttribute("mu_comp", str)
    lambda_components = G.attachEdgeAttribute("lambda_comp", str)
    
    K = round(2*E/N)
    
    #first make sure K is even
    if K % 2 == 1:
        K+=1
    
    print(K)
    
    #First construct simple ring
    for x in range(1, N):
        G.addEdge(x, x - 1)
    G.addEdge(0, N-1)
    
    #add in neighbor connections to form ring lattice
    for site in G.iterNodes():
        for i in range(1, K//2):
            G.addEdge(site, (site+i)%N)
            #G.addEdge(site, site-i)
    G.removeMultiEdges()
    for site in G.iterNodes():
        for j in range(site+1, (site+(K//2))%N):
            #visualize(G)
            if np.random.random() < beta:
                k=-1
                while k == -1 or k == site:
                    k = np.random.randint(0, N)
                #print(f"{site}, {j%N}")
                G.removeEdge(site, j%N)
                G.addEdge(site, k)
    G.removeMultiEdges()
    for site in G.iterNodes():
        #using the factor of E/N brings us closer to critical point
        healing_factor[site] = np.random.random()*(E/N)
        is_active[site] = 0
        components[site] = f"{site}"
        mu_components[site] = f"n{site}"
        
    G.removeMultiEdges()
    for edge in G.iterEdges():
        lambda_val = scipy.stats.powerlaw.rvs(disorder_alpha, size=1)[0]
        G.setWeight(edge[0], edge[1], lambda_val)
        #edge = (site, other_site)
        eid = G.edgeId(edge[0], edge[1])
        edge_components[eid] = f"{edge}"
        lambda_components[eid] = f"e({edge[0]},{edge[1]})"
    
    
    
    
    
    #G.indexEdges()
        
    if viz:
        visualize(G)
    
    return G
       


def gen_with_seed(L, seed):
    np.random.seed(seed)
    G = generate_square_lattice(L, L)
    return G
        
def generate_complete(n, visualize_on=False, disorder_alpha = 1):
    #Self explanatory
    G = nk.graph.Graph(n=n, weighted = True, edgesIndexed=True)
    
    for node in G.iterNodes():
        for other_node in G.iterNodes():
            if node !=other_node:
                G.addEdge(node, other_node)
    
    G.removeMultiEdges()
    
    if visualize_on:
        print("generated; now visualizing")
        visualize(G)
    
    
    """if disordered_lambda:
        nk.graphtools.randomizeWeights(G) #these weights act as lambda vals
    else:
        for u,v in G.iterEdges():
            G.setWeight(u,v,constant_lambda)"""
    
    #init activity values (infected)
    is_active = G.attachNodeAttribute("active", int)
    infection_progenitor = G.attachNodeAttribute("inf_prog", int)
    for site in G.iterNodes():
        infection_progenitor[site] = site
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    
    #node & edge components have the *like* components that have merged to form any given edge
    components = G.attachNodeAttribute("components", str) 
    edge_components = G.attachEdgeAttribute("e_comp", str)
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = G.attachNodeAttribute("mu_comp", str)
    lambda_components = G.attachEdgeAttribute("lambda_comp", str)
    
    #add disorder to healing values too if desired
    for u in G.iterNodes():
        #mu = scipy.stats.powerlaw.rvs(disorder_alpha, size=1)
        #print(mu)
        mu = np.random.random()
        healing_factor[u] = mu#[0]
        is_active[u] = 0
        components[u] = f"{u}"
        mu_components[u] = f"n{u}"
    
    for edge in G.iterEdges():
        lambda_val = scipy.stats.powerlaw.rvs(disorder_alpha, size=1)
        
        G.setWeight(edge[0], edge[1], lambda_val[0])
        eid = G.edgeId(edge[0], edge[1])
        edge_components[eid] = f"{edge}"
        lambda_components[eid] = f"e({edge[0]},{edge[1]})"
    print("done!")
        
    return G

def generate_power_law_log_mean(N, alpha, target_value):
    xmin = np.exp(target_value - 1/(alpha - 1))
    
    # Generate N samples
    u = np.random.uniform(0, 1, N)
    x = xmin * (1 - u) ** (-1/(alpha - 1))

    return x

def cut_torus_to_square(G_in):
    G = copy_graph(G_in)
    
    lambda_components = G.getEdgeAttribute("lambda_comp", str)
    edge_components = G.getEdgeAttribute("e_comp", str)
    
    L = int(math.sqrt(G.numberOfNodes()))
    top_bottom_vals = []
    side_vals = []
    #cut top/bottom
    for x in range(L):  
        this_node = x
        other_node = x + (L*(L-1))
        if G.hasEdge(this_node, other_node):
            eid = G.edgeId(this_node, other_node)
            
            vals = []
            vals.append(G.weight(this_node, other_node))
            vals.append(edge_components[eid])
            vals.append(lambda_components[eid])
            
            top_bottom_vals.append(vals)
            #remove_edge_safe(G, this_node, other_node)
            G.removeEdge(this_node, other_node)
        else:
            vals = [0, "", ""]
            #vals.append(G.weight(this_node, other_node))
            #vals.append(edge_components[eid])
            #vals.append(lambda_components[eid])
            top_bottom_vals.append(vals)
        
        #print(f"cut {this_node}-{other_node}")
    
    #cut sides
    for y in range(L):
        this_node = y*L
        other_node = this_node + L - 1
        if G.hasEdge(this_node, other_node):
            eid = G.edgeId(this_node, other_node)
            
            vals = []
            vals.append(G.weight(this_node, other_node))
            vals.append(edge_components[eid])
            vals.append(lambda_components[eid])
            
            side_vals.append(vals)
            #remove_edge_safe(G, this_node, other_node)
            G.removeEdge(this_node, other_node)
            #print(f"cut {this_node}-{other_node}")
        else:
            vals = [0, "", ""]
            side_vals.append(vals)
            
    #G.indexEdges(force=True)
    #print("done")
    return G, top_bottom_vals, side_vals

def get_graph_theta(G):
    h_list = []
    J_list = []
    h = G.getNodeAttribute("mu", float)
    for node in G.iterNodes():
        h_list.append(h[node])
    for weight in G.iterEdgesWeights():
        J_list.append(weight)
    
    lnh_bar = sum([math.log(i) for i in h_list])/len(h_list)
    lnJ_bar = sum([math.log(i[2]) for i in J_list])/len(J_list)
    
    #print(lnh_bar / lnJ_bar)
    #print(lnh_bar - lnJ_bar)
    
    return lnh_bar / lnJ_bar
    #return lnh_bar - lnJ_bar

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
    infection_progenitor = G.attachNodeAttribute("inf_prog", int)
    for site in G.iterNodes():
        infection_progenitor[site] = site
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
  
def convert_from_sophie_in(x, y, r):
    G = nk.graph.Graph(n=len(x), weighted=True, edgesIndexed=True)
    num_nodes = len(x)
    for node in range(num_nodes):
        #visualize(G)
        for other_node in range(num_nodes):
            if other_node != node:
                d = math.sqrt((x[node]-x[other_node])**2 + (y[node]-y[other_node])**2)
                if d < r[node]:
                    G.addEdge(node, other_node, w=1/d, checkMultiEdge=True)
                    
    #init activity values (infected)
    is_active = G.attachNodeAttribute("active", int)
    
    #init mu values
    healing_factor = G.attachNodeAttribute("mu", float)
    
    #node & edge components have the *like* components that have merged to form any given edge
    components = G.attachNodeAttribute("components", str) 
    edge_components = G.attachEdgeAttribute("e_comp", str)
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = G.attachNodeAttribute("mu_comp", str)
    lambda_components = G.attachEdgeAttribute("lambda_comp", str)
    
    #add disorder to healing values too if desired
    for u in G.iterNodes():
        mu = np.random.random()
        healing_factor[u] = mu
        is_active[u] = 0
        components[u] = f"{u}"
        mu_components[u] = f"n{u}"
            
    for edge in G.iterEdges():
        eid = G.edgeId(edge[0], edge[1])
        edge_components[eid] = f"{edge}"
        lambda_components[eid] = f"e({edge[0]},{edge[1]})"
    return G

def sophie_test():
    
    x = [200, 400]
    y = [150, 150]
    r = [10, 10]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [200, 400]
    y = [150, 150]
    r = [250, 250]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 300]
    y = [200, 200]
    r = [200, 200]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 200, 300, 100, 200, 300, 100, 200, 300]
    y = [100, 200, 300, 200, 100, 200, 300, 300, 100]
    r = [150, 300, 10, 150, 150, 10, 10, 10, 10]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    x = [100, 200, 300, 100, 200, 300, 100, 200, 300]
    y = [100, 200, 300, 200, 100, 200, 300, 300, 100]
    r = [150, 400, 150, 10, 10, 10, 150, 10, 150]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 200, 300, 100, 200, 300, 100, 200, 300]
    y = [100, 200, 300, 200, 100, 200, 300, 300, 100]
    r = [90, 300, 90, 150, 150, 150, 90, 150, 90]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 200, 300, 100, 200, 300, 100, 200, 300]
    y = [100, 200, 300, 200, 100, 200, 300, 300, 100]
    r = [110, 300, 110, 150, 150, 150, 110, 150, 110]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 150, 240, 290]
    y = [200, 200, 200, 200]
    r = [100, 60, 100, 60]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 150, 240, 290]
    y = [200, 200, 200, 200]
    r = [60, 60, 100, 60]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 200, 300, 300]
    y = [100, 100, 100, 300]
    r = [120, 120, 120, 250]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [200, 400, 200, 400]
    y = [200, 200, 400, 400]
    r = [250, 250, 250, 250]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 200, 500]
    y = [100, 100, 100]
    r = [110, 110, 110]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 200, 150]
    y = [100, 100, 186.6]
    r = [110, 110, 110]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    x = [99, 100, 200]
    y = [300, 300, 300]
    r = [99, 99, 199]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [200, 300, 400]
    y = [200, 200, 200]
    r = [300, 300, 300]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 400, 700]
    y = [200, 200, 200]
    r = [350, 350, 350]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [300, 100, 300, 500, 300]
    y = [300, 300, 100, 300, 500]
    r = [210, 210, 210, 210, 210]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [100, 280, 460, 640, 820, 1000]
    y = [200, 200, 200, 200, 200, 200]
    r = [200, 200, 200, 200, 200, 200]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
    
    x = [200, 400, 450, 700, 850]
    y = [200, 200, 200, 200, 200]
    r = [220, 220, 220, 220, 220]
    
    G = convert_from_sophie_in(x, y, r)
    visualize(G)
    try:
        G = sdrg_to_completion(G, visualizeSteps=True, verbose=True)
    except:
        print("no connections")    
 
""" ------------------------------------ GRAPH SPARSIFICATION ------------------------------------ """

######### FINDING CRITICAL POINT

def find_critical_mu_mod(G_in, num_iterations, samples_per_iteration, sample_t_max, init_mod = 8.0):
    mu_mod = init_mod
    delta_mu = 0.1
    G = copy_graph(G_in)
    G = scale_mu(G, mu_mod)
    
    for i in range(num_iterations):
        sim_sum = 0
        last_step_all_died = False
        for sample in range(samples_per_iteration):
            reset_DCP(G)
            #returns true if sim survives, false if it enters absorbing state
            sim_survived = sparsified_DCP_fast_notracking(G, t_max = sample_t_max)
            if sim_survived:
                sim_sum += 1
        if sim_sum == 0:
            mu_mod *= 1-delta_mu
            G = scale_mu(G, 1-delta_mu)
            print(f"lowering mu to {mu_mod}, simsum = {sim_sum}")
        elif sim_sum/samples_per_iteration > 0.33:
            mu_mod *= 1+delta_mu
            G = scale_mu(G, 1+delta_mu)
            print(f"raising mu to {mu_mod} simsum = {sim_sum}")
        else:
            delta_mu *= 0.5
            mu_mod *= 1+delta_mu
            G = scale_mu(G, 1+delta_mu)
            print(f"halving delta and raising mu to {mu_mod}, simsum = {sim_sum}")
        #print(mu_mod)
    
    return mu_mod
            
                        
    
    """
    for i in range(num_iterations):
        print(mu_scale_factor)
        above_criticality = check_above_criticality(G)
        if above_criticality != last_step_was_above_criticality and i > 0:
            #print("crossed criticality ")
            other_side_of_last_crossing = mu_scale_factor
            delta_mu *= 0.5
            crossed_criticality_at_least_once = True
            
        if above_criticality:
            #print("above criticality")
            G = scale_mu(G, 1 - delta_mu)
            mu_scale_factor *= 1-delta_mu
            
        else:   
            #print("below criticality")
            #mu_scale_factor += delta_mu
            G = scale_mu(G, 1 + delta_mu)
            mu_scale_factor *= 1+delta_mu
        last_step_was_above_criticality = above_criticality
    """


def find_variance():
    #G = copy_graph(G_in)
    G = generate_square_lattice(16, 16, disorder_alpha=4)
    a = SDRG_crit_point_estimation(G, 25)
    
    print(a)
    
    crit_points = []
    
    for i in range(50):
        print(f" ---- Trial #{i} ----")
        trial_crit_point = find_critical_mu_mod(G, 15, 15, 256, init_mod = a)
        crit_points.append(trial_crit_point)
    plt.hist(crit_points, bins=10)
    print("----------")
    print(crit_points)
    
          
def check_above_criticality(G_in, iteration=0):
    #returns False if there IS a correlated node/nodes
    #returns True if there are NO correlated nodes; i.e, the replica correlation function = 0; i.e, mu is too high
    G = copy_graph(G_in)
    
    L_sq = int(G.numberOfNodes()/2)
    
    clusters = get_neil_output(G, verbose = False)
    #vis_given_clusters(clusters, int(math.sqrt(L_sq)), "lattice", f"Iteration #{iteration}") # This doesn't work bc the clusters are 
    #print(clusters)
    print("checking...")
    for node in range(L_sq):
        cluster_node_is_in = get_sublist_with_value(node, clusters)
        #print(cluster_node_is_in)
        if (node + (L_sq)) in cluster_node_is_in:
            #print(cluster_node_is_in)
            #print(node)
            #print(node + L_sq)
            return False
        
    return True

def scale_mu(G_in, mu_scale):
    G = copy_graph(G_in)
    healing_factor = G.getNodeAttribute("mu", float)
    for node in G.iterNodes():
        healing_factor[node] = mu_scale*healing_factor[node]
    return G


def scale_lambda(G_in, lambda_scale):
    G = copy_graph(G_in)
    healing_factor = G.getNodeAttribute("mu", float)
    for edge in G.iterEdges():
        G.setWeight(edge[0], edge[1], G_in.weight(edge[0], edge[1])*lambda_scale)
    return G

      
def test4(G_in):
    #final_mu_scale = 1.0
    mu_scale_factor = 7.0
    G_in = scale_mu(G_in, mu_scale_factor)
    L = int(math.sqrt(G_in.numberOfNodes()))
    
    G0, vert_vals, horiz_vals = cut_torus_to_square(G_in)
    G1, vert_vals, horiz_vals = cut_torus_to_square(G_in)
    #visualize(G0)
    #visualize(G1)
    G = append_graphs(G0, G1)
    #visualize(G)
    #print(G.numberOfNodes())
    lambda_components = G.getEdgeAttribute("lambda_comp", str)
    edge_components = G.getEdgeAttribute("e_comp", str)
    for x in range(L):  
        if horiz_vals[x][0] > 0:
            e0 = x
            e1 = x + (((2*L) - 1)*L)
            G.addEdge(e0, e1, w=horiz_vals[x][0])
            eid = G.edgeId(e0, e1)
            edge_components[eid] = horiz_vals[x][1]
            lambda_components[eid] = horiz_vals[x][2]
            
            e0 = x + ((L-1)*L)
            e1 = x + ((L-1)*L) + L
            G.addEdge(e0, e1, w=horiz_vals[x][0])
            eid = G.edgeId(e0, e1)
            edge_components[eid] = horiz_vals[x][1]
            lambda_components[eid] = horiz_vals[x][2]
        
    for y in range(L):
        if vert_vals[y][0] > 0:
            right_side_node = ((y*L) + (L - 1))
            left_side_node = (y*L)
            
            e0 = right_side_node
            e1 = right_side_node + (L*L) - L + 1
            G.addEdge(e0, e1, w=vert_vals[y][0])
            eid = G.edgeId(e0, e1)
            edge_components[eid] = vert_vals[x][1]
            lambda_components[eid] = vert_vals[x][2]
            
            e0 = left_side_node
            e1 = left_side_node + (L*L) + L - 1
            G.addEdge(e0, e1, w=vert_vals[y][0])
            eid = G.edgeId(e0, e1)
            edge_components[eid] = vert_vals[x][1]
            lambda_components[eid] = vert_vals[x][2]
            
    G_sdrg = sdrg_sparsify(G)

def SDRG_crit_point_estimation(G_in, num_iterations, delta_mu = 0.1):
    #final_mu_scale = 1.0
    mu_scale_factor = 1.0
    G_in = scale_mu(G_in, mu_scale_factor)
    L = int(math.sqrt(G_in.numberOfNodes()))
    
    G0, vert_vals, horiz_vals = cut_torus_to_square(G_in)
    G1, vert_vals, horiz_vals = cut_torus_to_square(G_in)
    #visualize(G0)
    #visualize(G1)
    G = append_graphs(G0, G1)
    #visualize(G)
    #print(G.numberOfNodes())
    lambda_components = G.getEdgeAttribute("lambda_comp", str)
    edge_components = G.getEdgeAttribute("e_comp", str)
    for x in range(L):  
        if horiz_vals[x][0] > 0:
            e0 = x
            e1 = x + (((2*L) - 1)*L)
            G.addEdge(e0, e1, w=horiz_vals[x][0])
            eid = G.edgeId(e0, e1)
            edge_components[eid] = horiz_vals[x][1]
            lambda_components[eid] = horiz_vals[x][2]
            
            e0 = x + ((L-1)*L)
            e1 = x + ((L-1)*L) + L
            G.addEdge(e0, e1, w=horiz_vals[x][0])
            eid = G.edgeId(e0, e1)
            edge_components[eid] = horiz_vals[x][1]
            lambda_components[eid] = horiz_vals[x][2]
        
    for y in range(L):
        if vert_vals[y][0] > 0:
            right_side_node = ((y*L) + (L - 1))
            left_side_node = (y*L)
            
            e0 = right_side_node
            e1 = right_side_node + (L*L) - L + 1
            G.addEdge(e0, e1, w=vert_vals[y][0])
            eid = G.edgeId(e0, e1)
            edge_components[eid] = vert_vals[x][1]
            lambda_components[eid] = vert_vals[x][2]
            
            e0 = left_side_node
            e1 = left_side_node + (L*L) + L - 1
            G.addEdge(e0, e1, w=vert_vals[y][0])
            eid = G.edgeId(e0, e1)
            edge_components[eid] = vert_vals[x][1]
            lambda_components[eid] = vert_vals[x][2]
    
    print("doubling and gluing complete; beginning optimization")
    
    #visualize(G)
    #print_graph_values(G)
    #above_criticality = check_above_criticality(G)
    last_step_was_above_criticality = True
    
    crossed_criticality_at_least_once = False
    other_side_of_last_crossing = 0
    
    for i in range(num_iterations):
        print(mu_scale_factor)
        above_criticality = check_above_criticality(G, i)
        if above_criticality != last_step_was_above_criticality and i > 0:
            #print("crossed criticality ")
            other_side_of_last_crossing = mu_scale_factor
            delta_mu *= 0.5
            crossed_criticality_at_least_once = True
            
        if above_criticality:
            #print("above criticality")
            G = scale_mu(G, 1 - delta_mu)
            mu_scale_factor *= 1-delta_mu
            
        else:   
            #print("below criticality")
            #mu_scale_factor += delta_mu
            G = scale_mu(G, 1 + delta_mu)
            mu_scale_factor *= 1+delta_mu
        last_step_was_above_criticality = above_criticality
    
    print("-------")
    if not crossed_criticality_at_least_once:
        print("never crossed criticality; network is an unknown distance from criticality")
    else:
        print(f"critical mu scaling is between: {other_side_of_last_crossing} - {mu_scale_factor}  ")
        print(f"interval has size: {abs(mu_scale_factor-other_side_of_last_crossing)}")
    return mu_scale_factor
            
def max_susceptibility_crit_point_estimation(G_in, min_mu, max_mu, steps, averaging=True):
    G = copy_graph(G_in)
    N = G.numberOfNodes()
    
    screening_steps = round(steps*0.75)
    honing_steps = round(steps*0.25)
    screening_linspace = np.linspace(min_mu, max_mu, num=screening_steps)
    
    chi_arr = []
    mu_arr = []
    for i in screening_linspace:
        G_sample = copy_graph(G)
        G_sample = scale_mu(G_sample, i)
        
        chi_sample_arr = []
        
        samples_per_sample = 3 if averaging else 1
        
        for j in range(samples_per_sample):
        
            data, densities = fast_dcp_until_quasistationary_memsafe(G_sample, viz=False, original_graph_size=G.numberOfNodes(), spearman_thresh=0.98, return_density_set = True)
            
            print("----- got data and densities!")
            
            #densities_list = np.array(densities)
            
            ev_rho = densities[0] #np.average(densities_list)
            
            ev_rho_sq = densities[1] #np.average(np.square(densities_list))
            
            chi_sample = N*( (ev_rho_sq - (ev_rho**2)) / ev_rho )
            chi_sample_arr.append(chi_sample)
        chi = np.average(chi_sample_arr)
        
        print(f"chi = {chi} at mu = {i}")
        chi_arr.append(chi)
        mu_arr.append(i)
        
        
    if max(chi_arr) == chi_arr[-1]:
        print("could not find maximum of susceptibility! increase your max_mu value to get a more accurate estimate")
        return max_mu
    if max(chi_arr) == chi_arr[0]:
        print("could not find maximum of susceptibility! lower your min_mu value to get a more accurate estimate")
        return min_mu
    
    lower_chi_max_bound = mu_arr[np.argmax(chi_arr)-1]
    upper_chi_max_bound = mu_arr[np.argmax(chi_arr)+1]
    
    print("Completed screening; started honing process")
    
    honing_linspace = np.linspace(lower_chi_max_bound, upper_chi_max_bound, num=honing_steps)
    
    #chi_arr = []
    #mu_arr = []
    for i in honing_linspace:
        G_sample = copy_graph(G)
        G_sample = scale_mu(G_sample, i)
        data, densities = fast_dcp_until_quasistationary_memsafe(G_sample, viz=False, original_graph_size=G.numberOfNodes(), spearman_thresh=0.98, return_density_set = True)
        
        print("----- got data and densities!")
        
        densities_list = np.array(densities)
        
        ev_rho = ev_rho = densities[0]
        
        ev_rho_sq = densities[1]
        
        chi = N*( (ev_rho_sq - (ev_rho**2)) / ev_rho )
        
        print(f"chi = {chi} at mu = {i}")
        chi_arr.append(chi)
        mu_arr.append(i)
    
    max_chi_ind = np.argmax(chi_arr)
    best_mu = mu_arr[max_chi_ind]
    
    print(f"Completed honing process; max chi value found is chi={chi_arr[max_chi_ind]} at mu={best_mu}")
    print(chi_arr)
    print(mu_arr)
    return best_mu

def get_chi(G):
    G_sample = copy_graph(G)
    data, densities = fast_dcp_until_quasistationary_memsafe(G_sample, viz=False, original_graph_size=G.numberOfNodes(), spearman_thresh=0.98, return_density_set = True)
    
    print("----- got data and densities!")
    
    densities_list = np.array(densities)
    
    ev_rho = np.average(densities_list)
    
    ev_rho_sq = np.average(np.square(densities_list))
    
    chi = G.numberOfNodes()*( (ev_rho_sq - (ev_rho**2)) / ev_rho )
    
    print(f"chi = {chi}")

######### FUNDAMENTAL STEP


def sdrg_step(G, neil_mode = False, decimated_sites=[], logging_toggle = True, decimation_log = [], sparsify_mode = False, sparsify_log = [], partial=False, partial_dict=dict(), output_energy_clusters = False, energy_clusters=set(), keep_connected=False, keep_connected_set=set(), kawashima_filtering=True, local_maxima_filtering=False, modified_maximum_rule=False, step=0, visualizeStep = False, verbose=True):
    """This method does a single iteration of the SDRG sparsification. 
        - G: This is a networkit graph. It should be fully-connected/only contain one component, and needs to have edge weights + mu values (healing factor) + components (initialized with each node's own index)
        - neil_mode: Neil has a sparsification algorithm that outputs, at the end, all of the decimated clusters. This toggle just changes to output to this for checking the sdrg results against his
        - decimated_sites: only relevant in neil mode; this stores the sites that are decimated in each step as clusters
        - logging_toggle: This is my own logging; it is at time of writing, not used. Basically just an alternative to neil_mode logging
        - decimation_log: decimation_log is to logging_toggle as decimated_sites is to neil_mode. tracks the edges that get maximum rule-d out each step
        - sparsify_log: this contains all of the edges which were components of sites that were decimated
        - visualizeStep: visualizes the network AFTER applying the sdrg step using networkx. This gets reeeaaaalllllyyyy slow above like a hundred nodes or so
        - keep_connected: If true, this adds the components of the strongest coupling of each decimated site to the sparsification log when it gets decimated. This guarantees that the cluster represented by the site never becomes isolated/forms a new component when the graph is rebuilt.
        - energy_clusters: this tracks the energy clusters; ie, every time a site is decimated it outputs the set of mu_components that comprised the site's transverse field/mu value
    
    """
    t0 = time.time_ns()
    edge_components = G.getEdgeAttribute("e_comp", str)
    n_nodes = G.numberOfNodes()
    healing_factor = G.getNodeAttribute("mu", float)
    components = G.getNodeAttribute("components", str)
    is_active = G.getNodeAttribute("active", int)
    mu_components = G.getNodeAttribute("mu_comp", str)
    lambda_components = G.getEdgeAttribute("lambda_comp", str)
    #Stopping condition
    if n_nodes == 1:
        if verbose:
            print("Network is fully sparsified")
        last_site = [u for u in G.iterNodes()]

        decimated_sites = np.append(decimated_sites, components[last_site[0]])
        if output_energy_clusters:
            return G, energy_clusters
        elif sparsify_mode and not partial and not keep_connected:
            return G, sparsify_log
        elif sparsify_mode and partial and keep_connected:
            return G, sparsify_log, partial_dict, keep_connected_set
        elif sparsify_mode and not partial and keep_connected:
            return G, sparsify_log, keep_connected_set
        elif sparsify_mode and partial:
            return G, sparsify_log, partial_dict, set()
        elif logging_toggle and not partial:
            return G, decimation_log
        elif logging_toggle and partial:
            return G, decimation_log, partial_dict
        
        elif neil_mode:
            return G, decimated_sites
        
        return G
     
    list_of_maxima = []   
    #Put all mu and lambda values into a single np array so that we can easily choose the greatest value
    mu_arr = np.array([np.array([healing_factor[u], u]).T for u in G.iterNodes()])
    
    #IF ERROR HERE: lambda_arr will throw an error 
    lambda_arr = np.array([np.array([G.weight(u,v), u, v]).T for u,v in G.iterEdges()]) #[[0.1234, 0, 1], [0.2345, 0, 2], ...]
    #print(lambda_arr)     
    if not local_maxima_filtering:
        
        
        
        # we perform this check because in a disconnected network, there will come a time when there are no edges but more than 1 node
        if len(lambda_arr) > 0:
            max_mu_lambda_index = np.argmax(np.concat((mu_arr[:,0], lambda_arr[:,0])))
        else:
            max_mu_lambda_index = np.argmax(mu_arr[:,0])
        
        t1 = time.time_ns()
        
        #print((t1-t0)/1000000000)
        
        list_of_maxima.append(max_mu_lambda_index)
        
    else:
        
        possible_site_maxima = set(list(G.iterNodes()))
        possible_edge_maxima = set(list(G.iterEdges()))
        edge_list = list(G.iterEdges())
        for site in possible_site_maxima:
            if healing_factor[site] > max([weight[1] for weight in G.iterNeighborsWeights(site)]):
                mu_arr_indices = [int(i[1]) for i in mu_arr]
                #print(mu_arr_indices)
                site_index = int(np.where(np.array(mu_arr_indices)==site)[0][0])
                list_of_maxima.append(site_index)
                #print(f"node: {site}")
                for edge in G.iterNeighbors(site):
                    
                    possible_edge_maxima.discard(edge)
        for edge in possible_edge_maxima:
            #edge = possible_edge_maxima[edge_index]
            w = G.weight(edge[0], edge[1])
            
            if w > healing_factor[edge[0]] and w > healing_factor[edge[1]] and w == max([weight[1] for weight in G.iterNeighborsWeights(edge[0])]) and w == max([weight[1] for weight in G.iterNeighborsWeights(edge[1])]):
                index = 0
                for edge_ind in range(len(lambda_arr)):
                    if (lambda_arr[edge_ind][1] == edge[0] and lambda_arr[edge_ind][2] == edge[1]) or (lambda_arr[edge_ind][1] == edge[1] and lambda_arr[edge_ind][2] == edge[0]):
                        index = edge_ind
                        break
                #print(index)
                list_of_maxima.append(index + n_nodes)
                #print(lambda_arr[index])
                #print(f"edge: {edge} with w={w}")
    
    #print(list_of_maxima)
    
    for mu_lambda_index in list_of_maxima:
        
        if mu_lambda_index > n_nodes-1: #true iff biggest value is a lambda
            t0 = time.time_ns()
            
            edge_to_decimate = lambda_arr[mu_lambda_index-n_nodes, 1:] #np list of length 2
            
            #union the two sets containing neighbors of u and v
            # sets automatically remove the duplicates. This now contains only the nodes which are neighbors of/connected to either
            pair_neighborhood = {u for u in G.iterNeighbors(edge_to_decimate[0])} | {v for v in G.iterNeighbors(edge_to_decimate[1])}
    
            #give a name to each side of the edge
            u = edge_to_decimate[0]
            v = edge_to_decimate[1]
            omega = G.weight(u,v)
            
            #print(f"decimating edge {edge_to_decimate} with lambda = {G.weight(u,v)}") 
    
            #note that, as a result of this step, this sdrg algorithm creates new node indices up to 2x the original size of the graph. This is important for some methods like fast_random_choose()
            k = G.addNode() # returns new node id, so k = new node id
            
            #print(f"holy shit a bond decimation!!1!11!!! btwn {u} (mu={healing_factor[u]}) and {v} (mu={healing_factor[v]}) to form {k}")
            
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
            #print(f"holy shit a bond decimation!!1!11!!! btwn {u} and {v} to form {k} with components: {components[k]}")
            mu_components[k] = f"{mu_components[u]}_{mu_components[v]}_{lambda_components[G.edgeId(u,v)]}"
            is_active[k] = 0
            
            for neighbor in pair_neighborhood:
                #we are merging nodes u and v
                # i for each neighbor
                J_ui = G.weight(u, neighbor)
                J_vi = G.weight(v, neighbor)
                
                if J_ui > 0 or J_vi > 0:
                    #sometimes weird shit happens when an edge has ultra-low weight. We just avoid these edges
                    ratio = 0
                    """if J_ui > 1e-322 and J_vi > 1e-322:
                        ratio = abs(J_ui/J_vi)"""
                    
                    if (J_ui > J_vi*2) or (J_vi > J_ui*2):
                        new_edge_weight = max(J_ui, J_vi)
                    else:
                        new_edge_weight = J_ui + J_vi
                    G.addEdge(k, neighbor, new_edge_weight)
                    eid = G.edgeId(k, neighbor)
                    if J_ui == new_edge_weight:
                        #print(get_list_of_edge_components(G, vneid))
                        uneid = G.edgeId(u, neighbor)
                        lambda_components[eid] = lambda_components[uneid]
                    elif J_vi == new_edge_weight:
                        vneid = G.edgeId(v, neighbor)
                        lambda_components[eid] = lambda_components[vneid]
                    else: #
                        uneid = G.edgeId(u, neighbor)
                        vneid = G.edgeId(v, neighbor)
                        lambda_components[eid] = f"{lambda_components[uneid]}_{lambda_components[vneid]}"
                    
                    
                    if J_ui + J_vi > new_edge_weight: #checks that the maximum rule removes something; in other terms, that J_ui and J_vi both exist
                        uneid = G.edgeId(u, neighbor)
                        vneid = G.edgeId(v, neighbor)
                        if logging_toggle:
                            #keep track of the edge that was kept for some reason?
                            if J_ui == new_edge_weight:
                                #print(get_list_of_edge_components(G, vneid))
                                edge_comp_list = get_list_of_edge_components(G, uneid)
                                for i in edge_comp_list:
                                    decimation_log.append(i)
                            else:
                                edge_comp_list = get_list_of_edge_components(G, vneid)
                                for i in edge_comp_list:
                                    decimation_log.append(i)
                                #decimation_log.append((u,neighbor))
                        #if logging_toggle:
                        edge_components[eid] = f"{edge_components[uneid]}_{edge_components[vneid]}"
                            
                        if partial:
                            #keep track of the edge 
                            #print("partial log adding a new edge ...")
                            if J_ui == new_edge_weight:
                                #print(get_list_of_edge_components(G, vneid))
                                edge_comp_list = get_list_of_edge_components(G, vneid)
                                for i in edge_comp_list:
                                    #add to the edge's entry in the dict: the energy scale/omega value when it was decimated, the weight of the edge itself, and the multiplicity of the edge
                                    if omega < partial_dict[i][0]:
                                        partial_dict[i] = [omega, J_vi, partial_dict[i][2] + 1]
                                    
                                    #partial_set.add((i, omega, J_vi))
                            else:
                                edge_comp_list = get_list_of_edge_components(G, uneid)
                                for i in edge_comp_list:
                                    if omega < partial_dict[i][0]:
                                        partial_dict[i] = [omega, J_ui, partial_dict[i][2] + 1]
                                    #partial_set.add((i, omega, J_ui))
                                #decimation_log.append((u,neighbor))
                    elif J_ui + J_vi == new_edge_weight and J_ui != new_edge_weight and J_vi != new_edge_weight:
                        uneid = G.edgeId(u, neighbor)
                        vneid = G.edgeId(v, neighbor)
                        edge_components[eid] = f"{edge_components[uneid]}_{edge_components[vneid]}"
                    else:
                        #If the maximum rule removes nothing; i.e, the "neighbor" node was only connected to one of the u or v nodse, 
                        #G.addEdge(k, neighbor, new_edge_weight)
                        #eid = G.edgeId(k, neighbor)
                        #if logging_toggle:
                        if G.weight(u, neighbor) > 0:
                            edge_components[eid] = f"{edge_components[G.edgeId(u, neighbor)]}"
                        elif G.weight(v, neighbor) > 0:
                            edge_components[eid] = f"{edge_components[G.edgeId(v, neighbor)]}"
                        else:
                            print("Something very strange has happened...")
                else:
                    
                    """if G.hasEdge(u, neighbor):
                        uneid = G.edgeId(u, neighbor)
                        
                        edge_comp_list = get_list_of_edge_components(G, uneid)
                        for i in edge_comp_list:
                            if omega < partial_dict[i][0]:
                                partial_dict[i] = [omega, J_ui, partial_dict[i][2] + 1]
                        G.removeEdge(u, neighbor)
                    if G.hasEdge(v, neighbor):
                        vneid = G.edgeId(v, neighbor)
                        
                        edge_comp_list = get_list_of_edge_components(G, vneid)
                        for i in edge_comp_list:
                            if omega < partial_dict[i][0]:
                                partial_dict[i] = [omega, J_vi, partial_dict[i][2] + 1]
                        G.removeEdge(v, neighbor)"""
                    #remove_edge_safe(G, u, neighbor)
                    #remove_edge_safe(G, v, neighbor)
                    #G.removeEdge(u, neighbor)
                    #G.removeEdge(v, neighbor)
                    print("encountered an ultra low weight edge. Did nothing to it")
                                    
            #remove nodes at end. We had to wait b/c otherwise we can't calculate weights in loop
            G.removeNode(u)
            G.removeNode(v)
            t1 = time.time_ns()
            #print(f"lambda decim: {(t1-t0)/1000000000}")
            #print()
            
        else: #true iff biggest val is a mu
            #print("--")
            
            
            
            t0 = time.time_ns()
            step_to_print_after = 125    
            if step > step_to_print_after:
                print("---")
                t0 = time.time_ns()
            #since we put the mu_arr in front of the lambda_arr, we can directly access the mu_arr list with our mu_lambda_index without getting indexing errors
            site_to_decimate = mu_arr[mu_lambda_index,1]
            omega = mu_arr[mu_lambda_index,0]
            #print(f"decimating site {site_to_decimate} with mu = {healing_factor[site_to_decimate]}")
            #build a set with all of the neighbors
            #print(mu_components[site_to_decimate])
            sparsify_log.append(mu_components[site_to_decimate])
            
            if output_energy_clusters:
                energy_clusters.add(mu_components[site_to_decimate])
            
            if keep_connected and G.numberOfNodes() > 1:
                #print(list(G.iterNeighborsWeights(site_to_decimate)))
                #print(site_to_decimate)
                #if list(G.iterNeighborsWeights(site_to_decimate)) == []:
                #    visualize(G)
                strongly_connected_neighbor = sorted(list(G.iterNeighborsWeights(site_to_decimate)), key=lambda u: u[1], reverse=False)[0][0]
                
                eid = G.edgeId(site_to_decimate, strongly_connected_neighbor)
                comps = lambda_components[eid].split("_")
                for comp in comps:
                    if comp[0] == 'e':
                        #add the edges as tuples to this set so that only unique values stay
                        keep_connected_set.add(eval(comp[1:]))
                #sparsify_log.append(lambda_components[eid])
                #print(components[site_to_decimate])
                
                #if time.time_ns()%1000 == 0 or site_to_decimate == 87:
                    #print(f"{lambda_components[eid]}  -- {mu_components[site_to_decimate]} -- {site_to_decimate}, {strongly_connected_neighbor}")
                    #print(f"keeping edge btwn {site_to_decimate} and {strongly_connected_neighbor}")
            #print(omega)
            
            #print(f"0: {(time.time_ns()-t0)/1000000000}")
            #t0 = time.time_ns()
            
            neighborhood = {u for u in G.iterNeighbors(site_to_decimate)}
            #print(len(neighborhood))
            #I've forgotten why I did this, it seems kind of stupid but I don't really want to bother changing it right now. 
            #it can probably be replaced by neighbors_checked = set()
            neighbors_checked = {-1}
            neighbors_checked.remove(-1)
            if len(neighborhood) == 1 and partial:
                nc = neighborhood.copy()
                neighbor = nc.pop()
                
                neid = G.edgeId(neighbor, site_to_decimate)
                
                edge_comp_list = get_list_of_edge_components(G, neid)
                for i in edge_comp_list:
                    if omega < partial_dict[i][0]:
                        partial_dict[i] = [omega, G.weight(neighbor, site_to_decimate), partial_dict[i][2] + 1]
                    #partial_set.add((i, omega, G.weight(neighbor, site_to_decimate)))
            
            if step > step_to_print_after:
                print((time.time_ns()-t0)/1000000000)
                t0 = time.time_ns()
            #print(len(neighborhood))
            #print(f"inloop-{site_to_decimate}")
            
            #print(f"1: {(time.time_ns()-t0)/1000000000}")
            
            
            new_edges = set()
            #num_ops = 0
            #print(len(neighborhood))
            num_ops0=0
            #for i in range(len(neighborhood)):
                #num_ops0 += i
                
            #print(num_ops0)
            #t0 = time.time_ns()
            for neighbor in neighborhood:
                s = neighborhood - neighbors_checked # s is the set of unchecked neighbors. For high-degree networks this will be a significant speedup but probably not so for lattices
                s.remove(neighbor) #because we've already checked it. Also prevents a single neighbor from being double-counted
                if s: #... has any elements
                    for other_neighbor in s:
                        #t0 = time.time_ns()
                        if step > step_to_print_after:
                            print("-")
                            print((time.time_ns()-t0)/1000000000)
                            t0 = time.time_ns()
                        #num_ops+=1
                        # before change to sets it was this: np.delete(neighborhood, neighbor).delete(neighbors_checked): 
                        # i = decimated site 
                        # j, k = neighbors
        
                        J_jk = G.weight(neighbor, other_neighbor) # can be 0 if there was no link
                        #r_hi = -math.log(healing_factor[site_to_decimate])
                        #kappa_ik = -math.log(G.weight(site_to_decimate, other_neighbor))
                        #kappa_ij = -math.log(G.weight(neighbor, site_to_decimate))
                        
                        #this is a long expression so I make it a variable. It's really just lambda_ui*lambda_uj/mu_u
                        a = (G.weight(site_to_decimate, other_neighbor)*G.weight(neighbor, site_to_decimate))/healing_factor[site_to_decimate]
                        
                        new_edge_weight = max(J_jk, a)
                        
                        
                        
                        #returns whether the addition was successful (i.e, didn't make a multiedge)
                        add_success = G.addEdge(neighbor, other_neighbor, new_edge_weight, checkMultiEdge = True)
                        if not add_success:
                            G.setWeight(neighbor, other_neighbor, new_edge_weight)
                        new_edges.add((neighbor, other_neighbor))
                        #print(f"added edge between {neighbor} and {other_neighbor}")
                        
                        #this edge will always exist
                        noneid = G.edgeId(neighbor, other_neighbor) #neighbor-other neighbor edge i d
                        stdneid = G.edgeId(site_to_decimate, neighbor) #site to decimate-neighbor edge i d
                        stdoneid = G.edgeId(site_to_decimate, other_neighbor)
                        if step > step_to_print_after:
                            print((time.time_ns()-t0)/1000000000)
                            t0 = time.time_ns()
                        #print(f"1.1: {(time.time_ns()-t0)/1000000000}")
                        #t0 = time.time_ns()
                        if partial and J_jk > 0:
                            #J_jk > 0 means that there is already a connection between neighbor and other_neighbor, so maximum rule will apply
                            #print("partial log adding a new edge ...")
                            if J_jk == new_edge_weight:
                                #if J_jk is stronger, then both J_ui and J_uj get decimated
                                uieid = G.edgeId(site_to_decimate, other_neighbor)
                                ujeid = G.edgeId(site_to_decimate, neighbor)
                                #visualize(G)
                                #print(f"std = {site_to_decimate}, n = {neighbor}, on = {other_neighbor}")
                                #print(f"1.1.0: {(time.time_ns()-t0)/1000000000}")
                                #t0 = time.time_ns()
                                edge_comp_list_ui = get_list_of_edge_components(G, uieid)
                                #print(edge_comp_list_ui)
                                edge_comp_list_uj = get_list_of_edge_components(G, ujeid)
                                #print(f"l1= {len(edge_comp_list_ui) + len(edge_comp_list_uj)}")
                                #print(f"1.1.1: {(time.time_ns()-t0)/1000000000}")
                                #t0 = time.time_ns()
                                for i in edge_comp_list_ui:
                                    #add the edge, the energy scale/omega value when it was decimated, and the value of the edge itself
                                    if omega < partial_dict[i][0]:
                                        partial_dict[i] = [omega, G.weight(site_to_decimate, other_neighbor), partial_dict[i][2] + 1]
                                    #partial_set.add((i, omega, G.weight(site_to_decimate, other_neighbor)))
                                for i in edge_comp_list_uj:
                                    #add the edge, the energy scale/omega value when it was decimated, and the value of the edge itself
                                    if omega < partial_dict[i][0]:
                                        partial_dict[i] = [omega, G.weight(site_to_decimate, neighbor), partial_dict[i][2] + 1]
                                    #partial_set.add((i, omega, G.weight(site_to_decimate, neighbor)))
                                #print(f"1.1.2: {(time.time_ns()-t0)/1000000000}")
                                #t0 = time.time_ns()
                            else:
                                #if lambda_ui*lambda_uj/mu_u, then J_jk gets decimated
                                jkeid = G.edgeId(neighbor, other_neighbor)
                                
                                edge_comp_list = get_list_of_edge_components(G, jkeid)
                                #print(f"l2= {len(edge_comp_list)}")
                                for i in edge_comp_list:
                                    if omega < partial_dict[i][0]:
                                        partial_dict[i] = [omega, J_jk, partial_dict[i][2] + 1]
                                    #partial_set.add((i, omega, J_jk))
                                #print(f"1.1.1--: {(time.time_ns()-t0)/1000000000}")
                                #t0 = time.time_ns()
                        #print(f"1.2: {(time.time_ns()-t0)/1000000000}")
                        #t0 = time.time_ns()
                        if step > step_to_print_after:
                            print(f"m...{(time.time_ns()-t0)/1000000000}" )
                            t0 = time.time_ns()
                        
                        if logging_toggle:
                            
                            
                            if J_jk == new_edge_weight:
                                
                                edge_comp_list = get_list_of_edge_components(G, stdoneid)
                                for i in edge_comp_list:
                                    decimation_log.append(i)
                                edge_comp_list = get_list_of_edge_components(G, stdneid)
                                for i in edge_comp_list:
                                    decimation_log.append(i)
                                #decimation_log.append((site_to_decimate, other_neighbor))
                                #decimation_log.append((site_to_decimate, neighbor))
                                
                            elif J_jk > 0: # for this elif and the else, we are modifying the weight by the weights J_ui and J_vi, so add those
                                edge_comp_list = get_list_of_edge_components(G, noneid)
                                for i in edge_comp_list:
                                    decimation_log.append(i)    
                                #decimation_log.append((neighbor, other_neighbor))
                                
                                edge_components[(noneid)] = f"{edge_components[stdneid]}_{edge_components[stdoneid]}"
                            else:
                                edge_components[(noneid)] = f"{edge_components[stdneid]}_{edge_components[stdoneid]}"
                        else:
                            if J_jk != new_edge_weight:
                                
                                edge_components[(noneid)] = f"{edge_components[stdneid]}_{edge_components[stdoneid]}"
                            #print(len(edge_components[(noneid)]))
                            #print(len(f"{edge_components[stdneid]}_{edge_components[stdoneid]}".split("_")))
                        #print(f"1.3: {(time.time_ns()-t0)/1000000000}")
                        #t0 = time.time_ns()
                        if step > step_to_print_after:
                            print((time.time_ns()-t0)/1000000000)
                            t0 = time.time_ns()
                        
                        if a == new_edge_weight: # J_ij * J_ik / h_i
                            lambda_components[noneid] = f"{lambda_components[stdneid]}_{lambda_components[stdoneid]}_{mu_components[site_to_decimate]}"
                        if step > step_to_print_after:
                            print((time.time_ns()-t0)/1000000000)
                            t0 = time.time_ns()
                        #print(f"1.4: {(time.time_ns()-t0)/1000000000}")
                        #t0 = time.time_ns()
                #The neighbor that just looped through all of the other neighbors will
                # have had all of its connections made and calculated, so no reason to 
                # do anything to it for the rest of the loop. This just preserves the 
                # action done in the line with s.remove(neighbor) for future loops
                neighbors_checked.add(neighbor)
            
            #print(num_ops)
            #print(f"2: {(time.time_ns()-t0)/1000000000}")
            #print(f"2.1: {((time.time_ns()-t0)/1000000000)/num_ops0}")
            #t0 = time.time_ns()
            
            if kawashima_filtering and partial:
                #we modify the kawashima filtering so that we only filter out edges connected to local maxima when doing a partial sparsification
                #loop through every newly-generated edge
                for edge in new_edges:
                    # a majorating triangle is formed when, for an edge (i,j) there exists a site k s.t. ln(J_ij) < ln(J_ik); ln(J_ij) < ln(J_jk); and ln(J_ij) < ln(J_ik) + ln(J_jk) - ln(h_k)
                    
                    i = edge[0]
                    j = edge[1]
                    #print(f"on edge {i}, {j}")
                    
                    J_ij = G.weight(i,j)
                    
                    #print(get_list_of_edge_components(G, eid))
                    #print(eid)
                    if J_ij > 0:
                                       
                        # & gives an intersection of sets; pair_neighborhood has site which neighbor both i and j
                        pair_neighborhood = set([k for k in G.iterNeighbors(i)]) & set([k for k in G.iterNeighbors(j)])

                        #for each one, check whether it forms a majorating triangle with any of the neighbors of the site_to_decimate
                        for site in pair_neighborhood:
                            J_ik = G.weight(i,site)
                            J_jk = G.weight(j,site)
    
                            #we could combine this and the next if into one, but this helps readability and also
                            # acts as a sort of "short-circuit" like in matlab b/c many edges will fail this test
                            if J_ij < J_jk and J_ij < J_ik and J_ij < (J_jk*J_ik)/healing_factor[site]:
                                #comps = get_list_of_edge_components(G, G.edgeId(i, j))
                                #for 
                                greater_site = i if healing_factor[i] > healing_factor[j] else j
                                
                                #this if is fulfilled iff the greater of the two sites i,j, is a local maximum
                                
                                if healing_factor[greater_site] > max([weight[1] for weight in G.iterNeighborsWeights(greater_site)]):
                                    
                                    eid = G.edgeId(i, j)
                                    
                                    edge_comp_list = get_list_of_edge_components(G, eid)
                                    
                                    for val in edge_comp_list:
                                        #add the edge, the energy scale/omega value when it was decimated, and the value of the edge itself
                                        if healing_factor[greater_site] < partial_dict[val][0]:
                                            partial_dict[val] = [healing_factor[greater_site], J_ij, partial_dict[val][2] + 1]
                                        #partial_set.add((i, omega, G.weight(site_to_decimate, other_neighbor)))
                                    
                                    G.removeEdge(i,j)
                                    break
                            """
                            if J_jk > 1e-320 and J_ik > 1e-320 and J_ij > 1e-320 and healing_factor[site] > 1e-320:
                                if ln(J_ij) < ln(J_jk) and ln(J_ij) < ln(J_ik) and ln(J_ij) < ln(J_jk) + ln(J_ik) - ln(healing_factor[site]):
                                    #if so, delete it and break
                                    #print(f"removing edge between {i} and {j}")
                                    G.removeEdge(i,j)
                                    break
                            elif J_jk < 1e-320 or J_ik < 1e-320 and J_ij > 1e-320:
                                G.removeEdge(i,j)
                                break"""
                            #We don't need an else b/c any J_jk or J_ik value small enough to throw an error is too small to be greater than ln(J_ij)
                            #it's technically possible for J_ij to be super small too but I think it will probably never happen
                            #re the above comment; it DOES sometimes happen. 
            elif kawashima_filtering:
                #we modify the kawashima filtering so that we only filter out edges
                #loop through every newly-generated edge
                for edge in new_edges:
                    # a majorating triangle is formed when, for an edge (i,j) there exists a site k s.t. ln(J_ij) < ln(J_ik); ln(J_ij) < ln(J_jk); and ln(J_ij) < ln(J_ik) + ln(J_jk) - ln(h_k)
                    
                    i = edge[0]
                    j = edge[1]
                    #print(f"on edge {i}, {j}")
                    
                    J_ij = G.weight(i,j)
                    
                    #print(get_list_of_edge_components(G, eid))
                    #print(eid)
                    if J_ij > 0:
                                       
                        # & gives an intersection of sets; pair_neighborhood has site which neighbor both i and j
                        pair_neighborhood = set([k for k in G.iterNeighbors(i)]) & set([k for k in G.iterNeighbors(j)])

                        #for each one, check whether it forms a majorating triangle with any of the neighbors of the site_to_decimate
                        for site in pair_neighborhood:
                            J_ik = G.weight(i,site)
                            J_jk = G.weight(j,site)
    
                            if J_ij < J_jk and J_ij < J_ik and J_ij < (J_jk*J_ik)/healing_factor[site]:
                                #comps = get_list_of_edge_components(G, G.edgeId(i, j))
                                #for 
                                G.removeEdge(i,j)
                                break
                
            #print(num_ops)
            #print((time.time_ns()-t0)/1000000000)
            #t0 = time.time_ns()
            decimated_sites = np.append(decimated_sites, components[site_to_decimate])
            #print(f"decimated site {site_to_decimate} with components {components[site_to_decimate]}")
            if logging_toggle:
                #decimation_log.append([len(decimation_log), site_to_decimate, healing_factor[site_to_decimate]])
                pass
            #finally we get to actually remove the node
            #print(mu_components[site_to_decimate])
            #print(completeness(G))
            #visualize(G)
            G.removeNode(site_to_decimate)
            
            t1 = time.time_ns()
            #print(f"mu decim: {(t1-t0)/1000000000}")
        
    t2 = time.time_ns()
    #print(f"{(t1-t0)/1000000000}s for 0-1; {(t2-t1)/1000000000} for 1-2; {max_mu_lambda_index > n_nodes-1}")
    
    #nk.graphio.writeGraph(G, f"network_{time.time_ns()}_T.gml", nk.Format.GML)
    if visualizeStep:
        visualize(G)
    if output_energy_clusters:
        return G, energy_clusters
    elif sparsify_mode and not partial and not keep_connected:
        return G, sparsify_log
    elif sparsify_mode and partial and keep_connected:
        return G, sparsify_log, partial_dict, keep_connected_set
    elif sparsify_mode and not partial and keep_connected:
        return G, sparsify_log, keep_connected_set
    elif sparsify_mode and partial:
        return G, sparsify_log, partial_dict, set()
    elif logging_toggle and not partial:
        return G, decimation_log
    elif logging_toggle and partial:
        return G, decimation_log, partial_dict
    
    elif neil_mode:
        return G, decimated_sites
    
    return G


def sdrg_step_partial(G, neil_mode = False, decimated_sites=[], logging_toggle = True, decimation_log = [], sparsify_mode = True, sparsify_log = [], keep_connected=False, visualizeStep = False, verbose=True):
    """This method does a single iteration of the SDRG sparsification. 
        - G: This is a networkit graph. It should be fully-connected/only contain one component, and needs to have edge weights + mu values (healing factor) + components (initialized with each node's own index)
        - neil_mode: Neil has a sparsification algorithm that outputs, at the end, all of the decimated clusters. This toggle just changes to output to this for checking the sdrg results against his
        - decimated_sites: only relevant in neil mode; this stores the sites that are decimated in each step as clusters
        - logging_toggle: This is my own logging; it is at time of writing, not used. Basically just an alternative to neil_mode logging
        - decimation_log: decimation_log is to logging_toggle as decimated_sites is to neil_mode. tracks the edges that get maximum rule-d out each step
        - sparsify_log: this contains all of the edges which were components of sites that were decimated
        - visualizeStep: visualizes the network AFTER applying the sdrg step using networkx. This gets reeeaaaalllllyyyy slow above like a hundred nodes or so
        - keep_connected: If true, this adds the components of the strongest coupling of each decimated site to the sparsification log when it gets decimated. This guarantees that the cluster represented by the site never becomes isolated/forms a new component when the graph is rebuilt.
    """
    t0 = time.time_ns()
    edge_components = G.getEdgeAttribute("e_comp", str)
    n_nodes = G.numberOfNodes()
    healing_factor = G.getNodeAttribute("mu", float)
    components = G.getNodeAttribute("components", str)
    is_active = G.getNodeAttribute("active", int)
    mu_components = G.getNodeAttribute("mu_comp", str)
    lambda_components = G.getEdgeAttribute("lambda_comp", str)
    #Stopping condition
    if n_nodes == 1:
        if verbose:
            print("Network is fully sparsified")
        last_site = [u for u in G.iterNodes()]

        decimated_sites = np.append(decimated_sites, components[last_site[0]])
        if logging_toggle:
            return G, decimation_log
        elif sparsify_mode:
            return G, sparsify_log
        elif neil_mode:
            return G, decimated_sites
        return G
     
    #Put all mu and lambda values into a single np array so that we can easily choose the greatest value
    mu_arr = np.array([np.array([healing_factor[u], u]).T for u in G.iterNodes()])
    lambda_arr = np.array([np.array([G.weight(u,v), u, v]).T for u,v in G.iterEdges()])
            
    max_mu_lambda_index = np.argmax(np.concat((mu_arr[:,0], lambda_arr[:,0])))
    
    t1 = time.time_ns()
    
    #print((t1-t0)/1000000000)
    
    if max_mu_lambda_index > n_nodes-1: #true iff biggest value is a lambda
        t0 = time.time_ns()
        edge_to_decimate = lambda_arr[max_mu_lambda_index-n_nodes, 1:] #np list of length 2
        
        #union the two sets containing neighbors of u and v
        # sets automatically remove the duplicates. This now contains only the nodes which are neighbors of/connected to either
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
        mu_components[k] = f"{mu_components[u]}_{mu_components[u]}_{lambda_components[G.edgeId(u,v)]}"
        is_active[k] = 0
        
        for neighbor in pair_neighborhood:
            #we are merging nodes u and v
            # i for each neighbor
            J_ui = G.weight(u, neighbor)
            J_vi = G.weight(v, neighbor)
            
            
            new_edge_weight = max(J_ui, J_vi)
            G.addEdge(k, neighbor, new_edge_weight)
            eid = G.edgeId(k, neighbor)
            if J_ui == new_edge_weight:
                #print(get_list_of_edge_components(G, vneid))
                uneid = G.edgeId(u, neighbor)
                lambda_components[eid] = lambda_components[uneid]
            else:
                vneid = G.edgeId(v, neighbor)
                lambda_components[eid] = lambda_components[vneid]
            
            if J_ui + J_vi > new_edge_weight: #checks that the maximum rule removes something; in other terms, that J_ui and J_vi both exist
                uneid = G.edgeId(u, neighbor)
                vneid = G.edgeId(v, neighbor)
                if logging_toggle:
                    if J_ui == new_edge_weight:
                        #print(get_list_of_edge_components(G, vneid))
                        edge_comp_list = get_list_of_edge_components(G, vneid)
                        for i in edge_comp_list:
                            decimation_log.append(i)
                    else:
                        edge_comp_list = get_list_of_edge_components(G, uneid)
                        for i in edge_comp_list:
                            decimation_log.append(i)
                        #decimation_log.append((u,neighbor))
                if logging_toggle:
                    edge_components[eid] = f"{edge_components[uneid]}_{edge_components[vneid]}"
            else:
                #G.addEdge(k, neighbor, new_edge_weight)
                #eid = G.edgeId(k, neighbor)
                if logging_toggle:
                    edge_components[eid] = f"{edge_components[G.edgeId(u, neighbor)]}" if G.weight(u, neighbor) > 0 else f"{edge_components[G.edgeId(v, neighbor)]}"
            
        #remove nodes at end. We had to wait b/c otherwise we can't calculate weights in loop
        G.removeNode(u)
        G.removeNode(v)
        t1 = time.time_ns()
        #print(f"lambda decim: {(t1-t0)/1000000000}")
        #print()
        
    else: #true iff biggest val is a mu
        t0 = time.time_ns()
        #since we put the mu_arr in front of the lambda_arr, we can directly access the mu_arr list with our max_mu_lambda_index without getting indexing errors
        site_to_decimate = mu_arr[max_mu_lambda_index,1]
        #print(f"decimating site {site_to_decimate} with mu = {healing_factor[site_to_decimate]}")
        #build a set with all of the neighbors
        #print(mu_components[site_to_decimate])
        sparsify_log.append(mu_components[site_to_decimate])
        
        if keep_connected:
            strongly_connected_neighbor = sorted(list(G.iterNeighborsWeights(site_to_decimate)), key=lambda u: u[1], reverse=False)[0][0]
            eid = G.edgeId(site_to_decimate, strongly_connected_neighbor)
            sparsify_log.append(lambda_components[eid])
            if time.time_ns()%1 == 0:
                print(f"{lambda_components[eid]}  -- {mu_components[site_to_decimate]} -- {site_to_decimate}, {strongly_connected_neighbor}")
        
        neighborhood = {u for u in G.iterNeighbors(site_to_decimate)}
        #print(len(neighborhood))
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
    
                    J_jk = G.weight(neighbor, other_neighbor) # can be 0 if there was no link
                    #r_hi = -math.log(healing_factor[site_to_decimate])
                    #kappa_ik = -math.log(G.weight(site_to_decimate, other_neighbor))
                    #kappa_ij = -math.log(G.weight(neighbor, site_to_decimate))
                    
                    #this is a long expression so I make it a variable. It's really just lambda_ui*lambda_uj/mu_u
                    a = (G.weight(site_to_decimate, other_neighbor)*G.weight(neighbor, site_to_decimate))/healing_factor[site_to_decimate]
                    
                    new_weight = max(J_jk, a)
                    
                    #returns whether the addition was successful (i.e, didn't make a multiedge)
                    add_success = G.addEdge(neighbor, other_neighbor, new_weight, checkMultiEdge = True)
                    if not add_success:
                        G.setWeight(neighbor, other_neighbor, new_weight)
                        
                    
                    #this edge will always exist
                    noneid = G.edgeId(neighbor, other_neighbor) #neighbor-other neighbor edge i d
                    stdneid = G.edgeId(site_to_decimate, neighbor) #site to decimate-neighbor edge i d
                    stdoneid = G.edgeId(site_to_decimate, other_neighbor)
                    
                    if logging_toggle:
                        
                        
                        if J_jk == new_weight:
                            
                            edge_comp_list = get_list_of_edge_components(G, stdoneid)
                            for i in edge_comp_list:
                                decimation_log.append(i)
                            edge_comp_list = get_list_of_edge_components(G, stdneid)
                            for i in edge_comp_list:
                                decimation_log.append(i)
                            #decimation_log.append((site_to_decimate, other_neighbor))
                            #decimation_log.append((site_to_decimate, neighbor))
                            
                        elif J_jk > 0: # for this elif and the else, we are modifying the weight by the weights J_ui and J_vi, so add those
                            edge_comp_list = get_list_of_edge_components(G, noneid)
                            for i in edge_comp_list:
                                decimation_log.append(i)    
                            #decimation_log.append((neighbor, other_neighbor))
                            
                            edge_components[(noneid)] = f"{edge_components[stdneid]}_{edge_components[stdoneid]}"
                        else:
                            edge_components[(noneid)] = f"{edge_components[stdneid]}_{edge_components[stdoneid]}"
                    
                    
                    if a == new_weight: # J_ij * J_ik / h_i
                        lambda_components[noneid] = f"{lambda_components[stdneid]}_{lambda_components[stdoneid]}_{mu_components[site_to_decimate]}"
                    
                    
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
        
        t1 = time.time_ns()
        #print(f"mu decim: {(t1-t0)/1000000000}")
        
    t2 = time.time_ns()
    #print(f"{(t1-t0)/1000000000}s for 0-1; {(t2-t1)/1000000000} for 1-2; {max_mu_lambda_index > n_nodes-1}")
    
    #nk.graphio.writeGraph(G, f"network_{time.time_ns()}_T.gml", nk.Format.GML)
    if visualizeStep:
        visualize(G)
    if logging_toggle:
        return G, decimation_log
    elif sparsify_mode:
        return G, sparsify_log
    elif neil_mode:
        return G, decimated_sites
    return G



def test7():
    L_list = [8, 16, 32]
    averages = []
    for L in L_list:
        n_edges_list_for_L = []
        for i in range(15):
            G = generate_square_lattice(L, L)
            gs = sdrg_sparsify(G, use_kawashima=True)
            n_edges = gs.numberOfEdges()
            n_edges_list_for_L.append(n_edges)
        print("----------------------")
        print(np.average(n_edges_list_for_L))
        averages.append(np.average(n_edges_list_for_L))
        print("----------------------")
    print(averages)
        

######### EDGE SDRG

def sdrg_to_completion(G, visualizeSteps = False, verbose=False, kawashima = True):
    sparsify_log = []
    if not verbose:
        orig_n_nodes = G.numberOfNodes()
        for i in range(G.numberOfNodes()):
            #t0 = time.time_ns()
            G, sparsify_log = sdrg_step(G, logging_toggle = False, sparsify_mode=True, sparsify_log=sparsify_log, visualizeStep=visualizeSteps, kawashima_filtering=kawashima)
            #t1 = time.time_ns()
            #print(sparsify_log)
            if i % 25 == 0:
                
                print(f"step {i}/{orig_n_nodes}; edge # is {G.numberOfEdges()}")
        #print(decimated_sites)
        
        print("--------------------------")
        #print(sparsify_log)
        return G, sparsify_log
    else:
        for i in range(G.numberOfNodes()):
            G, sparsify_log = sdrg_step_verbose(G, decimation_log=sparsify_log, visualizeStep=visualizeSteps)
        #print(decimated_sites)
        
        print("--------------------------")
            
        return G, sparsify_log


def sdrg_partial(G, visualizeSteps = False, verbose=False, keep_connected = True, use_kawashima=False, use_local_max_filtering=False, fully_sparse = False):
    if not fully_sparse:
        sparsify_log = []
        partial_dict=dict([((u, v), [0, 0, 0]) for u,v in G.iterEdges()]) # stored as {edgeId: [omega_decimated_at, weight, multiplicity]}
        keep_connected_set=set()
        num_nodes_originally = G.numberOfNodes()
        for i in range(G.numberOfNodes()):
            #t0 = time.time_ns()
            G, sparsify_log, partial_set, keep_connected_set = sdrg_step(
                G, 
                logging_toggle = False, 
                sparsify_mode=True, 
                sparsify_log=sparsify_log, 
                partial_dict=partial_dict, 
                visualizeStep=False, 
                keep_connected=keep_connected, 
                keep_connected_set=keep_connected_set, 
                partial=True, 
                kawashima_filtering=use_kawashima, 
                local_maxima_filtering=use_local_max_filtering,
                step=0
                )
            
            #t1 = time.time_ns()
            if i % 25 == 0:
                print(f"step {i}/{num_nodes_originally}; completeness is {completeness(G)*100}%")
        #print(decimated_sites)
        
        print("--------------------------")
            
        #partial_set = set([(edge, partial_dict[edge][0], partial_dict[edge][1]) for edge in partial_dict])
        
        return G, sparsify_log, partial_dict, keep_connected_set
    else:
        sparsify_log = []
        partial_dict=dict()
        keep_connected_set=set()
        num_nodes_originally = G.numberOfNodes()
        for i in range(G.numberOfNodes()):
            #t0 = time.time_ns()
            if keep_connected:
                G, sparsify_log, keep_connected_set = sdrg_step(
                    G, 
                    logging_toggle = False, 
                    sparsify_mode=True, 
                    sparsify_log=sparsify_log, 
                    partial_dict=partial_dict, 
                    visualizeStep=False, 
                    keep_connected=keep_connected, 
                    keep_connected_set=keep_connected_set, 
                    partial=False, 
                    kawashima_filtering=use_kawashima, 
                    local_maxima_filtering=use_local_max_filtering,
                    step=0
                    )
            else:
                G, sparsify_log = sdrg_step(
                    G, 
                    logging_toggle = False, 
                    sparsify_mode=True, 
                    sparsify_log=sparsify_log, 
                    partial_dict=partial_dict, 
                    visualizeStep=False, 
                    keep_connected=keep_connected, 
                    keep_connected_set=keep_connected_set, 
                    partial=False, 
                    kawashima_filtering=use_kawashima, 
                    local_maxima_filtering=use_local_max_filtering,
                    step=0
                    )
            
            #t1 = time.time_ns()
            if i % 25 == 0:
                print(f"step {i}/{num_nodes_originally}; completeness is {completeness(G)*100}%")
        #print(decimated_sites)
        
        print("--------------------------")
            
        return G, sparsify_log, keep_connected_set



def sdrg_sparsify(G, use_kawashima = True):
    G_tilde_1 = copy_graph(G)
    G_tilde_2 = copy_graph(G)
    mu_components = G_tilde_1.getNodeAttribute("mu_comp", str)
    print("running sdrg")
    G_tilde_1, sparsify_log = sdrg_to_completion(G_tilde_1, kawashima=use_kawashima)
    print("finished sdrg; building sparsified network")
    edges_to_include = []
    for node_id in G_tilde_1.iterNodes(): #we don't know the index of the last node, so this directly gets us the object
        list_of_components = mu_components[node_id].split("_")
        #print(list_of_components)
        for thing in list_of_components:
            if thing[0] == 'e':
                edges_to_include.append(eval(thing[1:]))
                
    #list_of_decimated_site_components = []
    #print(sparsify_log)
    for site_component in sparsify_log:
        component_list = site_component.split("_")
        #print(component_list)
        #clean_component = eval(site_component[1:]) if site_component[0] == 'n' else site_component
        #components = site_components.split("_")
        for component in component_list:
            if component[0] != 'n':
                edges_to_include.append(eval(component[1:]))
        #list_of_decimated_site_components.append(clean_component)
        
    """for thing in list_of_decimated_site_components:
        if thing[0] == 'e':
            edges_to_include.append(eval(thing[1:]))"""
    
    #print(edges_to_include)
    
    edges_to_include_set = set(edges_to_include)
    #print(edges_to_include_set)
    all_edges_set = set([edge for edge in G_tilde_2.iterEdges()])
    #print(all_edges_set)
    edges_to_remove = all_edges_set - edges_to_include_set
    #instead run sdrg to some energy scale then remove all edges from the decimation log?
    #print(decimation_log)
    #no_duplicate_decimation_log = list(dict.fromkeys(decimation_log))
    #print(edges_to_include_set)
    print(f"at end we'll have  {len(edges_to_include_set)} edges")
    for edge in edges_to_remove:
    
        #edge = no_duplicate_decimation_log[i]
        
        #print(f"removed {edge[0]}, {edge[1]} with weight {G_tilde_2.weight(edge[0], edge[1])}")
    
        G_tilde_2.removeEdge(int(edge[0]), int(edge[1]))
    print("done")
    return G_tilde_2


def sdrg_sparsify_partial(G, percent_to_keep, keep_network_connected, kawashima=False, local_max=False, return_incremental_sparsifications=False, increments = [], keep_connected_singly = False, use_multiplicity = False, get_minimal_sparsification = False):
    """
    Interesting parameter overview:
        kawashima: keep off; this changes the RG trajectory and 
                    typically builds a network that is very bad at its job
        local_max: generally keep on; this changes the RG trajectory to 
                    finish more quickly, ends with fewer edges, and doesn't 
                    change performance
        get_minimal_sparsification: will only work properly with 
                    'return_incremental_sparsification' turned on. Finds the 
                    sparsification with fewest edges (to within 1%) and adds it 
                    to the queue of graphs to make. Technically only checks up 
                    to 75%; I suspect one will never find a best sparsification 
                    beyond that point
    """
    
    G_tilde_1 = copy_graph(G)
    G_tilde_2 = copy_graph(G)
    G_tilde_3 = copy_graph(G)
    
    #flag for this particular type of sparsification b/c it runs way faster than others
    running_single_full_sparsification = not return_incremental_sparsifications and percent_to_keep == 0 and not get_minimal_sparsification
    
    #increments = np.linspace(0, 1, 11)

    mu_components = G_tilde_1.getNodeAttribute("mu_comp", str)
    print("running sdrg")
    
    t0 = time.time_ns()
    #long
    if not running_single_full_sparsification:
        #partial_set
        G_tilde_1, sparsify_log, partial_dict, keep_connected_set = sdrg_partial(G_tilde_1, keep_connected=keep_network_connected, use_kawashima=kawashima, use_local_max_filtering=local_max)
    else:
        G_tilde_1, sparsify_log, keep_connected_set = sdrg_partial(G_tilde_1, keep_connected=keep_network_connected, use_kawashima=kawashima, use_local_max_filtering=local_max, fully_sparse=True)
    #t1 = time.time_ns()
    print((time.time_ns()-t0)/1000000000)
    #print("finished sdrg; building sparsified network")
    edges_to_include = []
    for node_id in G_tilde_1.iterNodes(): #we don't know the index of the last node, so this directly gets us the object
        list_of_components = mu_components[node_id].split("_")
        #print(node_id)
        #print(list_of_components)
        for thing in list_of_components:
            if thing[0] == 'e':
                edges_to_include.append(eval(thing[1:]))
                
    #list_of_decimated_site_components = []
    #print(sparsify_log)
    #print("----")
    for site_component in sparsify_log:
        
        #print(site_component)
        component_list = site_component.split("_")
        #print(component_list)
        #clean_component = eval(site_component[1:]) if site_component[0] == 'n' else site_component
        #components = site_components.split("_")
        for component in component_list:
            if component[0] != 'n':
                edges_to_include.append(eval(component[1:]))
        
        #list_of_decimated_site_components.append(clean_component)
        
    """for thing in list_of_decimated_site_components:
        if thing[0] == 'e':
            edges_to_include.append(eval(thing[1:]))"""
    #print(sorted(edges_to_include))
    edges_to_include_set = set(edges_to_include)
    all_edges_set = set([edge for edge in G_tilde_2.iterEdges()])
    
    #instead run sdrg to some energy scale then remove all edges from the decimation log?
    #print(decimation_log)
    #no_duplicate_decimation_log = list(dict.fromkeys(decimation_log))
    #print(edges_to_include_set)
    #t0 = time.time_ns()
    
    print("built sparse backbone; now adding partial edges")
    
    if running_single_full_sparsification:
        #in this case we just need to return the sdrg backbone without any partial edges. 
        #so just run the keep_connected procedure
        
        edges_to_remove = all_edges_set - edges_to_include_set    
        for edge in edges_to_remove:
        
            #edge = no_duplicate_decimation_log[i]
            
            #print(f"removed {edge[0]}, {edge[1]} with weight {G_tilde_2.weight(edge[0], edge[1])}")
        
            G_tilde_2.removeEdge(int(edge[0]), int(edge[1]))    
        
        if keep_network_connected:
            #to keep the network fully connected, we:
            print("Construction completed; now adding in connections to keep network fully connected")
            #find the components
            cc = nk.components.ConnectedComponents(G_tilde_2)
            cc.run()
            #print(f"{len(cc.getComponentSizes())} components: {cc.getComponents()}")
            #print(max(cc.getComponentSizes().values()))
            
            #format them into a nice list so that we can easily search through
            components_list = sorted(cc.getComponents(), key=lambda x: len(x), reverse=True)
            print("-")
            components_list = components_list[1:]
            #minimum_required_connection_edges = set()
            for component in components_list:
                for site in component:
                    #for every site, in every component, add in the (unique, b/c this goes into a set) edges which were the strongest at the time of component decimation 
                    required_edges = {edge for edge in keep_connected_set if site in edge}
                    if not keep_connected_singly:
                        for edge in required_edges:
                            edges_to_include_set.add(edge)
                    else:
                        strongest_edge_connected_to_component = max(required_edges, key=lambda edge: G.weight(edge[0], edge[1]))
                        edges_to_include_set.add(strongest_edge_connected_to_component)
            
            #print(components_list)
            
            edges_to_remove = all_edges_set - edges_to_include_set    
            for edge in edges_to_remove:
            
                #edge = no_duplicate_decimation_log[i]
                
                #print(f"removed {edge[0]}, {edge[1]} with weight {G_tilde_2.weight(edge[0], edge[1])}")
            
                G_tilde_3.removeEdge(int(edge[0]), int(edge[1]))    
                
            
            t1 = time.time_ns()
            #print(len(partial_set))
            print((t1-t0)/1000000000)
            return G_tilde_3
        return G_tilde_2
        
        
    #After this, we are NOT running a single full sparsification
    #print(len(partial_set))
    #first prune the edges which are already included in the SDRG backbone
    already_included_edges = set()
    edges_to_be_readded = set()
    
    #the following two for loops run in O(N) where N = number of edges in original network
    
    #remove the edges already in the backbone from the partial dict
    for edge in edges_to_include_set:#edges_to_include_set = {(u0, v0), (u1, v1), ...}
        del partial_dict[edge]    
        #partial_dict.pop(edge)
    
    #put all of the other edges in the partial dict into the set of edges to be readded
    for edge in partial_dict:
        edges_to_be_readded.add(edge)
    
    """for key in partial_dict:
        # triplet for the edge between u and v is: ((u,v), omega_edge_uv_got_decimated_at, weight_of_edge_uv)
        
        if key in edges_to_include_set:
            already_included_edges.add(key)
            #print(triplet)
        else:
            #print(tuple(sorted(triplet[0])))
            #print(sorted(triplet[0]))
            edges_to_be_readded.add(tuple(sorted(triplet[0])))
    for triplet in already_included_edges:
        partial_set.remove(triplet)"""
    #print(len(partial_set))
    print(f"The SDRG backbone has {len(edges_to_include_set)} edges")
    #long
    #print(len(edges_to_be_readded))
    
    
        
    #now, partial_set has all of the edges which got removed, but with some being duplicated at multiple energy levels
    #to fix this, we search through and find the lowest energy scale that each got removed at
    
    edge_multiplicity_dict = {edge: 0 for edge in G_tilde_2.iterEdges()}
    if use_multiplicity:
        for edge in edges_to_be_readded:
            edge_multiplicity_dict[edge] = partial_dict[edge][2]
            
    #if use_multiplicity:
        #print("---max & min multiplicity---")
        #print(max(edge_multiplicity_dict.values()))
        #print(min(edge_multiplicity_dict.values()))
        #multipliers = np.linspace(0.5, 2, G_tilde_3.numberOfEdges())
        
    #we reverse so that the edges which lasted the *least* amount of time (i.e, were fully decimated at the *highest* energy scale) are readded first
    partial_edges_sorted = sorted([[edge, partial_dict[edge][0], partial_dict[edge][1], partial_dict[edge][2]] for edge in partial_dict], key=lambda x: (x[1], x[2]), reverse=True) #sorted([triplet for triplet in partial_set], key=lambda x: (x[1], x[2]), reverse=True)
    #print(partial_edges_sorted)
    
    #attempt to find the best (minimal edge count) possible sparsification:
    def check_num_edges_for_certain_sparsification_percentage(G_in, partial_edges_sorted, keep_connected_set, num_edges_to_readd, all_edges, included_edges, keep_network_connected):
        G = copy_graph(G_in)
        Gt2 = copy_graph(G)
        Gt3 = copy_graph(G)
        
        all_edges_set = all_edges.copy()
        edges_to_include_set = included_edges.copy()
        for i in range(num_edges_to_readd):#"""round(len(partial_edges_sorted)*percentage)"""
            edges_to_include_set.add(partial_edges_sorted[i][0])
        
        edges_to_remove = all_edges_set - edges_to_include_set    
        for edge in edges_to_remove:
        
            #edge = no_duplicate_decimation_log[i]
            
            #print(f"removed {edge[0]}, {edge[1]} with weight {G_tilde_2.weight(edge[0], edge[1])}")
        
            Gt2.removeEdge(int(edge[0]), int(edge[1]))    
        
        cc = nk.components.ConnectedComponents(Gt2)
        cc.run()
        #print(f"{len(cc.getComponentSizes())} components: {cc.getComponents()}")
        #print(max(cc.getComponentSizes().values()))
        
        #format them into a nice list so that we can easily search through
        components_list = sorted(cc.getComponents(), key=lambda x: len(x), reverse=True)
        #print("-")
        components_list = components_list[1:]
        #minimum_required_connection_edges = set()
        for component in components_list:
            for site in component:
                #for every site, in every component, add in the (unique, b/c this goes into a set) edges which were the strongest at the time of component decimation 
                required_edges = {edge for edge in keep_connected_set if site in edge}
                
                for edge in required_edges:
                    edges_to_include_set.add(edge)
                
                    
        edges_to_remove = all_edges_set - edges_to_include_set    
        for edge in edges_to_remove:
        
            #edge = no_duplicate_decimation_log[i]
            
            #print(f"removed {edge[0]}, {edge[1]} with weight {G_tilde_2.weight(edge[0], edge[1])}")
        
            Gt3.removeEdge(int(edge[0]), int(edge[1]))
        return Gt3.numberOfEdges()
        #for i in range(0,50):
    
    
    def find_sparsification_percentage_closest_to_number_of_edges(target_edge_number, G, G_tilde_2, partial_edges_sorted, keep_connected_set, all_edges_set, edges_to_include_set, keep_network_connected):
        min_distance = G_tilde_2.numberOfEdges()
        closest_percentage = 0
        for i in range(len(partial_edges_sorted)):
            #p = i/100
            edge_count = check_num_edges_for_certain_sparsification_percentage(G_tilde_2, partial_edges_sorted, keep_connected_set, i, all_edges_set, edges_to_include_set, keep_network_connected)
            if abs(edge_count-target_edge_number) <= min_distance:
                min_distance = abs(edge_count-target_edge_number)
                closest_percentage = i/G.numberOfEdges()
                
        print(f"closest percentage is {closest_percentage*100}%, which is {min_distance} edges away from the target value")
                
        return closest_percentage
    
    if get_minimal_sparsification:
        print("Finding lowest edge count sparsification")
        
        min_edge_count = G_tilde_2.numberOfEdges()
        min_edge_count_percentage = 0
        for i in range(len(partial_edges_sorted)):
            #p = i/100
            edge_count = check_num_edges_for_certain_sparsification_percentage(G_tilde_2, partial_edges_sorted, keep_connected_set, i, all_edges_set, edges_to_include_set, keep_network_connected)
            if edge_count <= min_edge_count:
                min_edge_count = edge_count
                min_edge_count_percentage = i/G.numberOfEdges()
        increments = np.append(increments, min_edge_count_percentage)
        
        percent_to_keep = min_edge_count_percentage
        
        print(f"Found best sparsification; it has {min_edge_count} edges with {min_edge_count_percentage*100}% of partial edges added ")
    
    if not return_incremental_sparsifications:
        print("=================")
        print("Building a single sparse network")
        
        increments = [percent_to_keep]
    
    #convert our target percentages to real values
    if get_minimal_sparsification:
        #if we use the minimal sparsification then we should skip the last value in increments b/c it was already calculated
        for i in range(len(increments)-1):
            target_edge_number = round(increments[i]*G.numberOfEdges())
            corresponding_target_percentage = find_sparsification_percentage_closest_to_number_of_edges(target_edge_number, G, G_tilde_2, partial_edges_sorted, keep_connected_set, all_edges_set, edges_to_include_set, keep_network_connected)
            increments[i] = corresponding_target_percentage
    else:
        #otherwise we can just go through every value in increments
        for i in range(len(increments)):
            target_edge_number = round(increments[i]*G.numberOfEdges())
            print(target_edge_number)
            
            corresponding_target_percentage = find_sparsification_percentage_closest_to_number_of_edges(target_edge_number, G, G_tilde_2, partial_edges_sorted, keep_connected_set, all_edges_set, edges_to_include_set, keep_network_connected)
            
            increments[i] = corresponding_target_percentage
    
    
    
    list_of_sparsified_networks = []
    
    for sparsification_level in increments:
        G_tilde_2 = copy_graph(G)
        G_tilde_3 = copy_graph(G)
        edges_to_include_set = set(edges_to_include)
        
        for i in range(round(len(partial_edges_sorted)*sparsification_level)):
            edges_to_include_set.add(partial_edges_sorted[i][0])
        
        
        
        #print(f"at end we'll have  {len(edges_to_include_set)} edges")
        
        edges_to_remove = all_edges_set - edges_to_include_set    
        for edge in edges_to_remove:
        
            #edge = no_duplicate_decimation_log[i]
            
            #print(f"removed {edge[0]}, {edge[1]} with weight {G_tilde_2.weight(edge[0], edge[1])}")
        
            G_tilde_2.removeEdge(int(edge[0]), int(edge[1]))    
        
        if keep_network_connected:
            print("Construction completed; now adding in connections to keep network fully connected")
            cc = nk.components.ConnectedComponents(G_tilde_2)
            cc.run()
            #print(f"{len(cc.getComponentSizes())} components: {cc.getComponents()}")
            #print(max(cc.getComponentSizes().values()))
            components_list = sorted(cc.getComponents(), key=lambda x: len(x), reverse=True)
            #print("-")
            components_list = components_list[1:]
            #minimum_required_connection_edges = set()
            for component in components_list:
                for site in component:
                    required_edges = {edge for edge in keep_connected_set if site in edge}
                    if not keep_connected_singly:
                        for edge in required_edges:
                            edges_to_include_set.add(edge)
                    else:
                        if len(required_edges) > 0:
                            strongest_edge_connected_to_component = max(required_edges, key=lambda edge: G.weight(edge[0], edge[1]))
                            edges_to_include_set.add(strongest_edge_connected_to_component)
                        
            
            #print(components_list)
            
            edges_to_remove = all_edges_set - edges_to_include_set    
            for edge in edges_to_remove:
            
                #edge = no_duplicate_decimation_log[i]
                
                #print(f"removed {edge[0]}, {edge[1]} with weight {G_tilde_2.weight(edge[0], edge[1])}")
            
                G_tilde_3.removeEdge(int(edge[0]), int(edge[1]))    
        
            if use_multiplicity:
                max_mult = max(edge_multiplicity_dict.values())
                min_mult = min(edge_multiplicity_dict.values())
                #print(max_mult)
                #print(min_mult)
                for edge in G_tilde_3.iterEdges():
                    multiplier = ((edge_multiplicity_dict[edge] - min_mult)/(max_mult - min_mult))+0.5 #outputs number between 0.5 and 1.5 to multiply by
                    G_tilde_3.setWeight(edge[0], edge[1], G_tilde_3.weight(edge[0], edge[1]) * multiplier)

            
            t1 = time.time_ns()
            #print(len(partial_set))
            #print((t1-t0)/1000000000)
            list_of_sparsified_networks.append(G_tilde_3)
            print(f"ended up with {G_tilde_3.numberOfEdges()} edges")   
            #return G_tilde_3
        else:
            
            if use_multiplicity:
                max_mult = max(edge_multiplicity_dict.values())
                min_mult = min(edge_multiplicity_dict.values())
                #print(max_mult)
                #print(min_mult)
                for edge in G_tilde_2.iterEdges():
                    multiplier = ((edge_multiplicity_dict[edge] - min_mult)/(max_mult - min_mult))+0.5 #outputs number between 0.5 and 1.5 to multiply by
                    G_tilde_2.setWeight(edge[0], edge[1], G_tilde_2.weight(edge[0], edge[1]) * multiplier)

            
            t1 = time.time_ns()
            #print(len(partial_set))
            #print((t1-t0)/1000000000)
            list_of_sparsified_networks.append(G_tilde_2)
            #return G_tilde_2
            print(f"ended up with {G_tilde_2.numberOfEdges()} edges")        
    print((time.time_ns()-t0)/1000000000)
    if not return_incremental_sparsifications:
        return list_of_sparsified_networks[0]
    elif not get_minimal_sparsification:
        return list_of_sparsified_networks
    else:
        return list_of_sparsified_networks, increments[-1]
       

def sdrg_sparsify_n_edges(G, n_edges):
    G_tilde_1 = copy_graph(G)
    G_tilde_2 = copy_graph(G)
    
    G_tilde_1, decimation_log = sdrg_to_completion(G_tilde_1)
    
    #instead run sdrg to some energy scale then remove all edges from the decimation log?
    #print(decimation_log)
    no_duplicate_decimation_log = list(dict.fromkeys(decimation_log))
    print(len(no_duplicate_decimation_log))
    print(f"at end we'll have  {n_edges} edges")
    for i in range(int(G.numberOfNodes() - n_edges)):
    
        edge = no_duplicate_decimation_log[i]
        
        #print(f"removed {edge[0]}, {edge[1]} with weight {G_tilde_2.weight(edge[0], edge[1])}")
    
        G_tilde_2.removeEdge(int(edge[0]), int(edge[1]))
    print("done with sdrg sparsification")
    return G_tilde_2
    
def shortest_tree_sdrg(G_in, target_edge_count):
    """
    This method uses a sort of combination of SMDS and SDRG, where instead of 
    keeping ALL shortest paths like in the SMDS backbone, we keep only the 
    shortest paths which lead to a site in one of the largest clusters 
    identified by SDRG. I figured out that this might be a good idea after 
    looking at a quasistationary graph and realizing that probability seemed to 
    emanate out from the sites in the largest 3 clusters as if they were 
    "generators" of activity
    
    
    """
    G = copy_graph(G_in)
    G_dists = copy_graph(G)
    
    k = 3
    n_edges = 0
    
    num_clusters = get_num_clusters(G)
    
    while n_edges < target_edge_count:
        #n = G.numberOfNodes()
        
        k_cluster_sites = get_nodes_of_largest_k_clusters(G, k)
        set_of_k_cluster_sites = set(k_cluster_sites)
        all_sites = set([site for site in G.iterNodes()])
        
        non_cluster_sites = all_sites-set_of_k_cluster_sites
        
        #D = np.ones((n, n))*np.inf
        
        #B_matrix = np.zeros((n, n))
        for edge in G.iterEdges():
            d = (1/G.weight(edge[0], edge[1]))-1
            G_dists.setWeight(edge[0], edge[1], d)
            
            #print(f"adding {d} to {edge[0]}, {edge[1]} in D")
            #D[edge[0], edge[1]] = d
        
        #all pairs shortest path = APSP    
        APSP_solver = nk.distance.APSP(G_dists)
        APSP_solver.run()
        DTm = APSP_solver.getDistances(asarray=True)
        #print(DTm)
        #print(k_cluster_sites)
        sdrg_backbone = sdrg_sparsify_partial(G, 0, False, local_max=False)
        #visualize(sdrg_backbone)
        edges_to_keep = set([edge for edge in sdrg_backbone.iterEdges()])
        #print(edges_to_keep)
        
        for site in non_cluster_sites:
            dist_to_each_cluster_site = []
            for cluster_site in k_cluster_sites:
                dist_to_each_cluster_site.append(DTm[site][cluster_site]) #DTm is symmetric so row and column are interchangeable
            
            #this then gets us the site in the top k clusters which is closest to our non-cluster site
            nearest_cluster_site = k_cluster_sites[np.argmin(dist_to_each_cluster_site)]
            
            shortest_path_finder = nk.distance.Dijkstra(G, source=site, target=nearest_cluster_site, storePaths=True)
            shortest_path_finder.run()
            #nk.distance.Dijkstra(G, site, target=nearest_cluster_site).getPath()
            shortest_path_to_cluster = shortest_path_finder.getPath(nearest_cluster_site)
            #print("----")
            #print(f"going from: {site} -> {nearest_cluster_site}")
            #print(shortest_path_to_cluster)
            for i in range(len(shortest_path_to_cluster)-1):
                #add the whole path to the list of edges to keep
                #we sort this to ensure that edges don't get double added in opposite orders
                edge = tuple(sorted([shortest_path_to_cluster[i], shortest_path_to_cluster[i+1]]))
                edges_to_keep.add(edge)
        
        all_edges_set = set([edge for edge in G.iterEdges()])
        edges_to_remove = all_edges_set - edges_to_keep
        #print(edges_to_keep)
        #print(edges_to_remove)
        for edge in edges_to_remove:
            G.removeEdge(edge[0], edge[1])
            #print(f"removed edge from {edge[0]} to {edge[1]}")
        #visualize(G)
        if G.numberOfEdges() < target_edge_count and k < num_clusters/2:
            print("-------------------------------=============")
            print(G.numberOfEdges())
            print(k)
            
            k+=1
            
        else:
            print(f"FINISHED SHORTEST PATH AT K={k}")
            return G
    
    

######### CLUSTERING SDRG

def sdrg_to_omega(G, target_omega, visualizeSteps = False):
    decimation_log = []
    omega = get_max_omega(G)
    while omega > target_omega:
        G, decimation_log = sdrg_step(G, decimation_log=decimation_log, visualizeStep=visualizeSteps)
        omega = get_max_omega(G)
    
    print(f"sparsified to target omega {target_omega}; max omega at end is {omega}")
    print("--------------------------")
        
    return G

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



def sdrg_step_verbose(G, neil_mode = False, decimated_sites=[], logging_toggle = True, decimation_log = [], visualizeStep = False):
    """This method does a single iteration of the SDRG sparsification. 
        - G: This is a networkit graph. It should be fully-connected/only contain one component, and needs to have edge weights + mu values (healing factor) + components (initialized with each node's own index)
        - neil_mode: Neil has a sparsification algorithm that outputs, at the end, all of the decimated clusters. This toggle just changes to output to this for checking the sdrg results against his
        - decimated_sites: only relevant in neil mode; this stores the sites that are decimated in each step as clusters
        - logging_toggle: This is my own logging; it is at time of writing, not used. Basically just an alternative to neil_mode logging
        - decimation_log: decimation_log is to logging_toggle as decimated_sites is to neil_mode. tracks the edges that get maximum rule-d out each step
        - visualizeStep: visualizes the network AFTER applying the sdrg step using networkx. This gets reeeaaaalllllyyyy slow above like a hundred nodes or so
    """
    edge_components = G.getEdgeAttribute("e_comp", str)
    n_nodes = G.numberOfNodes()
    healing_factor = G.getNodeAttribute("mu", float)
    components = G.getNodeAttribute("components", str)
    is_active = G.getNodeAttribute("active", int)
    mu_components = G.getNodeAttribute("mu_comp", str)
    lambda_components = G.getEdgeAttribute("lambda_comp", str)
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
    #print(max_mu_lambda_index)
    print(f"mu values are: {mu_arr}")
    print(f"lambda values are: {lambda_arr}")
    
    if max_mu_lambda_index > n_nodes-1: #true iff biggest value is a lambda
        edge_to_decimate = lambda_arr[max_mu_lambda_index-n_nodes, 1:] #np list of length 2
        
        #union the two sets containing neighbors of u and v
        # sets automatically remove the duplicates. This now contains only the nodes which are neighbors of/connected to either
        pair_neighborhood = {u for u in G.iterNeighbors(edge_to_decimate[0])} | {v for v in G.iterNeighbors(edge_to_decimate[1])}

        #give a name to each side of the edge
        u = edge_to_decimate[0]
        v = edge_to_decimate[1]
        print(f"decimating edge {edge_to_decimate} with lambda = {G.weight(u,v)}") 

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
        mu_components[k] = f"{mu_components[u]}_{mu_components[u]}_{lambda_components[G.edgeId(u,v)]}"
        is_active[k] = 0
        
        for neighbor in pair_neighborhood:
            #we are merging nodes u and v
            # i for each neighbor
            J_ui = G.weight(u, neighbor)
            J_vi = G.weight(v, neighbor)
            
            
            new_edge_weight = max(J_ui, J_vi)
            G.addEdge(k, neighbor, new_edge_weight)
            eid = G.edgeId(k, neighbor)
            if J_ui == new_edge_weight:
                #print(get_list_of_edge_components(G, vneid))
                uneid = G.edgeId(u, neighbor)
                lambda_components[eid] = lambda_components[uneid]
            else:
                vneid = G.edgeId(v, neighbor)
                lambda_components[eid] = lambda_components[vneid]
            
            if J_ui + J_vi > new_edge_weight: #checks that the maximum rule removes something; in other terms, that J_ui and J_vi both exist
                uneid = G.edgeId(u, neighbor)
                vneid = G.edgeId(v, neighbor)
                if logging_toggle:
                    if J_ui == new_edge_weight:
                        #print(get_list_of_edge_components(G, vneid))
                        edge_comp_list = get_list_of_edge_components(G, vneid)
                        for i in edge_comp_list:
                            decimation_log.append(i)
                    else:
                        edge_comp_list = get_list_of_edge_components(G, uneid)
                        for i in edge_comp_list:
                            decimation_log.append(i)
                        #decimation_log.append((u,neighbor))
                
                edge_components[eid] = f"{edge_components[uneid]}_{edge_components[vneid]}"
            else:
                #G.addEdge(k, neighbor, new_edge_weight)
                #eid = G.edgeId(k, neighbor)
                edge_components[eid] = f"{edge_components[G.edgeId(u, neighbor)]}" if G.weight(u, neighbor) > 0 else f"{edge_components[G.edgeId(v, neighbor)]}"
            
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
    
                    J_jk = G.weight(neighbor, other_neighbor) # can be 0 if there was no link
                    #r_hi = -math.log(healing_factor[site_to_decimate])
                    #kappa_ik = -math.log(G.weight(site_to_decimate, other_neighbor))
                    #kappa_ij = -math.log(G.weight(neighbor, site_to_decimate))
                    
                    #this is a long expression so I make it a variable. It's really just lambda_ui*lambda_uj/mu_u
                    a = (G.weight(site_to_decimate, other_neighbor)*G.weight(neighbor, site_to_decimate))/healing_factor[site_to_decimate]
                    
                    new_weight = max(J_jk, a)
                    
                    #returns whether the addition was successful (i.e, didn't make a multiedge)
                    add_success = G.addEdge(neighbor, other_neighbor, new_weight, checkMultiEdge = True)
                    if not add_success:
                        G.setWeight(neighbor, other_neighbor, new_weight)
                        
                    
                    #this edge will always exist
                    noneid = G.edgeId(neighbor, other_neighbor) #neighbor-other neighbor edge i d
                    stdneid = G.edgeId(site_to_decimate, neighbor) #site to decimate-neighbor edge i d
                    stdoneid = G.edgeId(site_to_decimate, other_neighbor)
                    if logging_toggle:
                        
                        
                        if J_jk == new_weight:
                            edge_comp_list = get_list_of_edge_components(G, stdoneid)
                            for i in edge_comp_list:
                                decimation_log.append(i)
                            edge_comp_list = get_list_of_edge_components(G, stdneid)
                            for i in edge_comp_list:
                                decimation_log.append(i)
                            #decimation_log.append((site_to_decimate, other_neighbor))
                            #decimation_log.append((site_to_decimate, neighbor))
                            
                        elif J_jk > 0: # for this elif and the else, we are modifying the weight by the weights J_ui and J_vi, so add those
                            edge_comp_list = get_list_of_edge_components(G, noneid)
                            for i in edge_comp_list:
                                decimation_log.append(i)    
                            #decimation_log.append((neighbor, other_neighbor))
                            
                            edge_components[(noneid)] = f"{edge_components[stdneid]}_{edge_components[stdoneid]}"
                        else:
                            edge_components[(noneid)] = f"{edge_components[stdneid]}_{edge_components[stdoneid]}"
                    
                    if a == new_weight: # J_ij * J_ik / h_i
                        lambda_components[noneid] = f"{lambda_components[stdneid]}_{lambda_components[stdoneid]}_{mu_components[site_to_decimate]}"
                    
                    
            #The neighbor that just looped through all of the other neighbors will
            # have had all of its connections made and calculated, so no reason to 
            # do anything to it for the rest of the loop. This just preserves the 
            # action done in the line with s.remove(neighbor) for future loops
            neighbors_checked.add(neighbor)
            
        decimated_sites = np.append(decimated_sites, components[site_to_decimate])
        print(f"decimated site {site_to_decimate} with components {components[site_to_decimate]}")
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



def sampling_sparsification(G, n_samples, method = 'u', visualize_on = False):
    """
    THIS DOESN'T WORK
    Methods are uniform, weight-based, and effective resistance; corresponding to arguments of 'u', 'w', and 'er'
    """
    G_tilde = nk.graph.Graph(n=G.numberOfNodes(), weighted=True, edgesIndexed=True)
    edges = np.array([(u,v) for u,v in G.iterEdges()])
    
    if method == 'u':
        edge_pmf = np.ones(len(edges)) * (1/len(edges))
    elif method == 'w':
        pass

def threshold_sparsification(G, n_edges_left, viz = False):
    G_tilde = copy_graph(G) #nk.graph.Graph(n=G.numberOfNodes(), weighted=True, edgesIndexed=True)
    #G_edge_pmf = np.array([G.weight(u,v) for u,v in G.iterEdges()])*(1/sum([G.weight(u,v) for u,v in G.iterEdges()]))
    edges = sorted([(G.weight(u,v), u, v) for u,v in G.iterEdges()])
    #print(edges[0])
    #print(G_edge_pmf.size)
    
    for sample in range(G.numberOfEdges() - n_edges_left):
        #Simply remove the lowest edges up to a certain number of edges left
        
        G_tilde.removeEdge(edges[sample][1], edges[sample][2])
        
    
    if viz:
        visualize(G_tilde)
    return G_tilde



def uniform_sampling_sparsification(G, n_samples, visualize_on = False):
    #G_tilde = nk.graph.Graph(n=G.numberOfNodes(), weighted=True, edgesIndexed=True)
    G_tilde = copy_graph(G)
    edges = np.array([(u,v) for u,v in G.iterEdges()])
    list_of_edges_to_keep = []
    for sample in range(n_samples):
        choice_index = int(np.random.choice(np.arange(edges.size/2)))
        choice = edges[choice_index]
        list_of_edges_to_keep.append((choice[0], choice[1]))
        #we ensure that there are no duplicates
        
    #print(len(list_of_edges_to_keep))
    set_of_edges_to_keep = set(list_of_edges_to_keep)
    all_edge_set = set([edge for edge in G.iterEdges()])
    edges_to_remove = all_edge_set - set_of_edges_to_keep

    for edge in edges_to_remove:
        G_tilde.removeEdge(edge[0], edge[1])
    """for edge in G_tilde.iterEdges():
        if edge not in list_of_edges_to_keep:#and tuple(reversed(edge)) not in list_of_edges_to_keep:
            G_tilde.removeEdge(edge[0], edge[1])"""
    
    #print(G_tilde.numberOfEdges())
    
    #set_of_edges_to_keep = set(list_of_edges_to_keep)
    for edge in set_of_edges_to_keep:
        multiplicity = list_of_edges_to_keep.count(edge)
        G_tilde.setWeight(edge[0], edge[1], G.weight(edge[0], edge[1]) * multiplicity)
    
    if visualize_on:
        visualize(G_tilde)
    return G_tilde


def weight_sampling_sparsification(G, n_samples, visualize_on = False):
    #G_tilde = nk.graph.Graph(n=G.numberOfNodes(), weighted=True, edgesIndexed=True)
    G_tilde = copy_graph(G)
    G_edge_pmf = np.array([G.weight(u,v) for u,v in G.iterEdges()])*(1/sum([G.weight(u,v) for u,v in G.iterEdges()]))
    edges = np.array([(u,v) for u,v in G.iterEdges()])
    #print(edges[0])
    #print(G_edge_pmf.size)
    list_of_edges_to_keep = []
    for sample in range(n_samples):
        #Mercier et al. uses the probability p_e of choosing an edge in the new 
        # weight expression, so we get the index to ensure we can retreive the 
        # right probability from the edge pmf
        choice_index = int(np.random.choice(np.arange(edges.size/2), p=G_edge_pmf))
        #print(choice_index)
        choice = edges[choice_index]
        list_of_edges_to_keep.append((choice[0], choice[1]))
        
    set_of_edges_to_keep = set(list_of_edges_to_keep)
    all_edge_set = set([edge for edge in G.iterEdges()])
    edges_to_remove = all_edge_set - set_of_edges_to_keep

    for edge in edges_to_remove:
        G_tilde.removeEdge(edge[0], edge[1])
    
    """for edge in G_tilde.iterEdges():
        if edge not in list_of_edges_to_keep:
            G_tilde.removeEdge(edge[0], edge[1])"""
    
    #set_of_edges_to_keep = set(list_of_edges_to_keep)
    for edge in set_of_edges_to_keep:
        multiplicity = list_of_edges_to_keep.count(edge)
        G_tilde.setWeight(edge[0], edge[1], G.weight(edge[0], edge[1]) * multiplicity)
    
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
    
    #init activity values (infected)
    is_active = G_tilde.attachNodeAttribute("active", int)
    infection_progenitor = G_tilde.attachNodeAttribute("inf_prog", int)
    for site in G_tilde.iterNodes():
        infection_progenitor[site] = site
    #init mu values
    healing_factor = G_tilde.attachNodeAttribute("mu", float)
    components = G_tilde.attachNodeAttribute("components", str)
    
    edge_components = G_tilde.attachEdgeAttribute("e_comp", str)
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = G_tilde.attachNodeAttribute("mu_comp", str)
    lambda_components = G_tilde.attachEdgeAttribute("lambda_comp", str)
    healing_factor_G = G.getNodeAttribute("mu", float)
    
    for sample in range(n_samples):
        choice_index = int(np.random.choice(np.arange(E_list.size/2), p=G_edge_pmf))
        #print(choice_index)
        edge_to_sample = E_list[choice_index]
        #edge_to_sample = np.random.choice(E_list, p=G_edge_pmf)
        u = edge_to_sample[0]
        v = edge_to_sample[1]
        
        #G_tilde.addEdge(u,v, G.weight(u,v))
        if G_tilde.weight(u, v) == 0:
            new_weight = (G.weight(u, v) / (n_samples * G_edge_pmf[choice_index]))
        else:
            new_weight = G_tilde.weight(u, v) + (G.weight(u, v) / (n_samples * G_edge_pmf[choice_index]))
            G_tilde.removeEdge(u, v)
        
        G_tilde.addEdge(u, v, w=new_weight)
        new_eid = G_tilde.edgeId(u, v)
        edge_components[new_eid] = f"({u}, {v})"
        lambda_components[new_eid] = f"e({u}, {v})"
        #x = np.zeros(dim)
        #x[]
        
    
    for u in G_tilde.iterNodes():
        #mu = np.random.random()
        healing_factor[u] = healing_factor_G[u]
        is_active[u] = 0
        components[u] = f"{u}"
        mu_components[u] = f"n{u}"
    
        
    if visualize_on:
        visualize(G_tilde)
    
    return G_tilde
 

def semi_metric_synthetic_network_gen(G, target_tau_metric, viz = False):
    
    # B is the metric backbone
    B = nk.graph.Graph(n=G.numberOfNodes(), weighted = True, edgesIndexed = True)
    
    is_active = B.attachNodeAttribute("active", int)
    healing_factor = B.attachNodeAttribute("mu", float)
    components = B.attachNodeAttribute("components", str)
    infection_progenitor = B.attachNodeAttribute("inf_prog", int)
    for site in B.iterNodes():
        infection_progenitor[site] = site
    edge_components = B.attachEdgeAttribute("e_comp", str)
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = B.attachNodeAttribute("mu_comp", str)
    lambda_components = B.attachEdgeAttribute("lambda_comp", str)
    
    healing_factor_G = G.getNodeAttribute("mu", float)
    for u in B.iterNodes():
        healing_factor[u] = healing_factor_G[u]
        is_active[u] = 0
        components[u] = f"{u}"
        mu_components[u] = f"n{u}"
    
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

"""
def semi_metric_backbone(G, viz = False):
    
    # B is the metric backbone
    B = nk.graph.Graph(n=G.numberOfNodes(), weighted = True, edgesIndexed = True)
    
    edge_components = B.attachEdgeAttribute("e_comp", str)
    
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = B.attachNodeAttribute("mu_comp", str)
    lambda_components = B.attachEdgeAttribute("lambda_comp", str)
    
    is_active = B.attachNodeAttribute("active", int)
    healing_factor = B.attachNodeAttribute("mu", float)
    components = B.attachNodeAttribute("components", str)
    
    healing_factor_G = G.getNodeAttribute("mu", float)
    
    for u in B.iterNodes():
        healing_factor[u] = healing_factor_G[u]
        is_active[u] = 0
        components[u] = f"{u}"
        mu_components[u] = f"n{u}"
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
                
    for edge in B.iterEdges():
        eid = B.edgeId(edge[0], edge[1])
        edge_components[eid] = f"{edge}"
        lambda_components[eid] = f"e({edge[0]},{edge[1]})"
            
    if viz:
        visualize(B)
    
    
    return B

"""

def semi_metric_backbone(G, viz = False):
    B = SMDS_sparsifier(G, 0, viz=viz)
    return B

def SMDS_to_n_edges(G, n_edges, viz=False):
    
    # B is the metric backbone
    B = nk.graph.Graph(n=G.numberOfNodes(), weighted = True, edgesIndexed = True)
    
    is_active = B.attachNodeAttribute("active", int)
    healing_factor = B.attachNodeAttribute("mu", float)
    components = B.attachNodeAttribute("components", str)
    edge_components = B.attachEdgeAttribute("e_comp", str)
    infection_progenitor = B.attachNodeAttribute("inf_prog", int)
    for site in B.iterNodes():
        infection_progenitor[site] = site
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = B.attachNodeAttribute("mu_comp", str)
    lambda_components = B.attachEdgeAttribute("lambda_comp", str)
    
    healing_factor_G = G.getNodeAttribute("mu", float)
    for u in B.iterNodes():
        healing_factor[u] = healing_factor_G[u]
        is_active[u] = 0
        components[u] = f"{u}"
        mu_components[u] = f"n{u}"
    
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
                new_eid = B.edgeId(i, j)
                edge_components[new_eid] = f"({i}, {j})"
                lambda_components[new_eid] = f"e({i}, {j})"
                
                B_matrix[i,j] = G.weight(i,j)
                E_m += 1
            
            elif D[i,j] != np.inf:
                #keep track of the semi-metric edges 
                semi_metric_distortion = D[i,j]/DTm[i,j]
                semi_metric_edges.add((i, j, semi_metric_distortion))
                
                E_sm += 1
    
    print("Built metric backbone and found all semi-metric edges")
    
    sorted_semi_metric_edges = sorted(list(semi_metric_edges), key=lambda x: x[2])
    
    if n_edges < B.numberOfEdges():
        print("n={n_edges} is too small! The backbone has {B.numberOfEdges()} edges. Returning backbone!")
        
    else: 
        E_sm_to_add = int(n_edges - B.numberOfEdges())
                
        #E_sm_to_add = int(round(chi*E_sm))
        for i in range(E_sm_to_add):
            edge = sorted_semi_metric_edges[i]
            B.addEdge(edge[0], edge[1], w=G.weight(edge[0], edge[1]))
            
            #orig_eid = G.edgeId(edge[0], edge[1])
            new_eid = B.edgeId(edge[0], edge[1])
            edge_components[new_eid] = f"{edge}"
            lambda_components[new_eid] = f"e({edge[0]},{edge[1]})"
        
    if viz:
        visualize(B)
    
    
    return B

def SMDS_sparsifier(G, chi, output_num_components = False, viz = False):
    
    # B is the metric backbone
    B = nk.graph.Graph(n=G.numberOfNodes(), weighted = True, edgesIndexed = True)
    
    is_active = B.attachNodeAttribute("active", int)
    healing_factor = B.attachNodeAttribute("mu", float)
    components = B.attachNodeAttribute("components", str)
    edge_components = B.attachEdgeAttribute("e_comp", str)
    infection_progenitor = B.attachNodeAttribute("inf_prog", int)
    for site in B.iterNodes():
        infection_progenitor[site] = site
    #mu & lambda components have *all* components that have contributed to the mu or lambda values of a node/edge, respectively
    # these are stored as n123_e(34,42)_n53_ ..., with contributing nodes denoted n, and contributing edges denoted e
    mu_components = B.attachNodeAttribute("mu_comp", str)
    lambda_components = B.attachEdgeAttribute("lambda_comp", str)
    
    healing_factor_G = G.getNodeAttribute("mu", float)
    for u in B.iterNodes():
        healing_factor[u] = healing_factor_G[u]
        is_active[u] = 0
        components[u] = f"{u}"
        mu_components[u] = f"n{u}"
    
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
                new_eid = B.edgeId(i, j)
                edge_components[new_eid] = f"({i}, {j})"
                lambda_components[new_eid] = f"e({i}, {j})"
                
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
        
        #orig_eid = G.edgeId(edge[0], edge[1])
        new_eid = B.edgeId(edge[0], edge[1])
        edge_components[new_eid] = f"{edge}"
        lambda_components[new_eid] = f"e({edge[0]},{edge[1]})"
        
    if viz:
        visualize(B)
    
    if output_num_components:
        return B, number_of_components(B)
    return B


def get_neil_output(G_in, verbose = True, lm=False):
    G = copy_graph(G_in)
    decimated_sites = []
    if not lm:
        for i in range(G.numberOfNodes()):
            G, decimated_sites = sdrg_step(G, neil_mode = True, logging_toggle = False, sparsify_mode= False, decimated_sites=decimated_sites, visualizeStep=False, verbose=False, kawashima_filtering=True)
    else:
        for i in range(G.numberOfNodes()):
            G, decimated_sites = sdrg_step(G, neil_mode = True, logging_toggle = False, sparsify_mode= False, decimated_sites=decimated_sites, visualizeStep=False, verbose=False, kawashima_filtering=True, local_maxima_filtering=True)

    #print(decimated_sites)
    
    

    temp = [components.split("_") for components in decimated_sites]
    #print(temp)
    formatted_decimated_sites = [sorted([int(x) for x in sublist]) for sublist in temp]
    
    if verbose:
        print("--------------------------")
        print(formatted_decimated_sites)
    else:
        return formatted_decimated_sites
    
def get_energy_clusters(G_in, verbose = True, lm=False):
    G = copy_graph(G_in)
    energy_clusters = set()
    if not lm:
        for i in range(G.numberOfNodes()):
            #print("test")
            G, energy_clusters = sdrg_step(G, neil_mode = True, logging_toggle = False, sparsify_mode= False, visualizeStep=False, verbose=False, kawashima_filtering=True, energy_clusters=energy_clusters, output_energy_clusters=True)
    else:
        for i in range(G.numberOfNodes()):
            G, energy_clusters = sdrg_step(G, neil_mode = True, logging_toggle = False, sparsify_mode= False, visualizeStep=False, verbose=False, kawashima_filtering=True, local_maxima_filtering=True, energy_clusters=energy_clusters, output_energy_clusters=True)

    #print(decimated_sites)
    
    

    temp = [components.split("_") for components in energy_clusters]
    
    print(temp)
    formatted_decimated_sites = sorted([sorted([x for x in sublist]) for sublist in temp], key = lambda x: (len(x), x))
    just_nodes = []#[[] for cluster in formatted_decimated_sites]
    for cluster in formatted_decimated_sites:
        arr = []
        for element in cluster:
            if element[0] == 'n':
                arr.append(int(element[1:]))
        just_nodes.append(arr)
    if verbose:
        print("--------------------------")
        print(just_nodes)
    else:
        return just_nodes
    
def get_nodes_of_largest_k_clusters(G_in, k):
    clusters = get_neil_output(G_in, verbose=False)
    sorted_clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    nodes_to_output = []
    for i in range(k):
        cluster = sorted_clusters[i]
        for node in cluster:
            nodes_to_output.append(node)
    return nodes_to_output
    
def get_num_clusters(G_in):
    clusters = get_neil_output(G_in, verbose=False)
    
    return len(clusters)

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


def sparsified_DCP_fast_old(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100):
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
        print(f"{N_active} nodes active ------------------------------" )
        
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
    This differs from the original sparsified_DCP_fast bc it uses a slightly 
    different algorithm to run the CP, modified from 'Simulating the contact 
    process in heterogeneous environments' by Fallert et al., 2008. 
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
    

    Q_list = []
    for u in G.iterNodes():
        sum_neighbor_weights = sum([v[1] for v in G.iterNeighborsWeights(u)])
        #a = list(G.iterNeighborsWeights(u))
        Q_i = healing_factor[u] + sum_neighbor_weights
        Q_list.append(Q_i)
    #Q_list = [healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()]
    Q = max(Q_list)#2#max([healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()])
    

    while t < t_max:
        #t0=time.time_ns()
        
        N_active = len(active_nodes)
        #print(N_active)
        # the number of exposed nodes is the same as the number of edges connected to active sites
        #N_exposed = len(exposed_edges)
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check
        

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        #u3 = np.random.random()
        
        
        #t1=time.time_ns()
        #t2=0
        #t3=0
        #t4=0
        #t5=0
        deltaT = 0
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = mu_i / Q #(mu_i*N_active)/((mu_i*N_active)+N_exposed)
            neighbor_edge_weights = [u[1] for u in G.iterNeighborsWeights(random_site)]
            lambda_i = sum(neighbor_edge_weights)
            prob_to_infect = lambda_i / Q

            #t2=time.time_ns()
            #t4=time.time_ns()
            
            
            if u2 < prob_to_infect:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                exposed_sites = list(G.iterNeighbors(random_site))
                if len(exposed_sites) > 0:
                    #exposed_sites = list(G.iterNeighbors(random_site))#[site for site in G.iterNeighbors(random_site)]
                    #temp = [u[1] for u in G.iterNeighborsWeights(random_site)]
                    exposed_sites_pmf = np.array(neighbor_edge_weights)*(1/lambda_i)
                    site_to_infect = np.random.choice(exposed_sites, p=exposed_sites_pmf)
                    
                    if not is_active[site_to_infect]:
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
                        
                        
                    deltaT = 1/(Q*(N_active+1))
                    #t += deltaT
                    
                    #t5=time.time_ns()
                    #print("inf")
                    
            
            elif u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)

                is_active[random_site] = 0
                
                #newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                #for neighbor in G.iterNeighbors(random_site):
                #    if G.weight(random_site, neighbor) == lambda_max:
                #        exposed_edges.remove((random_site, neighbor))
                #        if len(exposed_edges) > 0:
                #            lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
                #        else:
                #            lambda_max = 0
                #            print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
                #            return transition_times
                #    else:
                #        exposed_edges.remove((random_site, neighbor))
                
                for neighbor in G.iterNeighbors(random_site):
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
                
                deltaT = 1/(Q*(N_active-1))
                #t += deltaT
                #t5=time.time_ns()
                #print("heal")
                
            else:
                #print("wasting")
                deltaT = 1/(Q*N_active)
            
            #vis_dcp(active_nodes, dimensions=[64, 64], title=f"dcp; t={t}")
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


def sparsified_DCP_fast_memsafe_s(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100):

    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    transition_times = {site : [0, 0, 0, False] for site in range(original_graph_size)}
    #          t_active, t_inactive, t_lastTransition, False=last transition was active->inactive

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
        
        #larger node indices appear after sparsification (even though there are fewer nodes), so this prevents out of bounds errors
        c = components[site].split("_")
        
        for component in c:
            transition_times[int(component)] = [0, 0, 0, True]
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from the original sparsified_DCP_fast bc it only tracks the 
    total times that each node is infected/healthy for instead of tracking 
    every transition. 
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
    

    Q_list = []
    for u in G.iterNodes():
        sum_neighbor_weights = sum([v[1] for v in G.iterNeighborsWeights(u)])
        #a = list(G.iterNeighborsWeights(u))
        Q_i = healing_factor[u] + sum_neighbor_weights
        Q_list.append(Q_i)
    #Q_list = [healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()]
    Q = max(Q_list)#2#max([healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()])
    

    while t < t_max:
        #t0=time.time_ns()
        
        N_active = len(active_nodes)
        #print(N_active)
        # the number of exposed nodes is the same as the number of edges connected to active sites
        #N_exposed = len(exposed_edges)
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check
        

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        #u3 = np.random.random()
        
        
        #t1=time.time_ns()
        #t2=0
        #t3=0
        #t4=0
        #t5=0
        deltaT = 0
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = mu_i / Q #(mu_i*N_active)/((mu_i*N_active)+N_exposed)
            neighbor_edge_weights = [u[1] for u in G.iterNeighborsWeights(random_site)]
            lambda_i = sum(neighbor_edge_weights)
            prob_to_infect = lambda_i / Q

            #t2=time.time_ns()
            #t4=time.time_ns()
            
            
            if u2 < prob_to_infect:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                exposed_sites = list(G.iterNeighbors(random_site))
                if len(exposed_sites) > 0:
                    #exposed_sites = list(G.iterNeighbors(random_site))#[site for site in G.iterNeighbors(random_site)]
                    #temp = [u[1] for u in G.iterNeighborsWeights(random_site)]
                    exposed_sites_pmf = np.array(neighbor_edge_weights)*(1/lambda_i)
                    site_to_infect = np.random.choice(exposed_sites, p=exposed_sites_pmf)
                    
                    if not is_active[site_to_infect]:
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
                            
                            transition_times[int(site)][1] += t - transition_times[int(site)][2]
                            transition_times[int(site)][2] = t
                            transition_times[int(site)][3] = True
                            #df = swap_column_vals_after_k(df, site_index, df_col)
                        
                        
                    deltaT = 1/(Q*(N_active+1))
                    #t += deltaT
            
            elif u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)

                is_active[random_site] = 0
                
                #newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                #for neighbor in G.iterNeighbors(random_site):
                #    if G.weight(random_site, neighbor) == lambda_max:
                #        exposed_edges.remove((random_site, neighbor))
                #        if len(exposed_edges) > 0:
                #            lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
                #        else:
                #            lambda_max = 0
                #            print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
                #            return transition_times
                #    else:
                #        exposed_edges.remove((random_site, neighbor))
                
                for neighbor in G.iterNeighbors(random_site):
                    exposed_edges.remove((random_site, neighbor))
                    
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                    
                #we could calculate df_col above "if N_active > 0" bc its global, but there are some wasted actions so may as well wait until we know we need it!
                #df_col = find_lt(seconds, t)
                for site in components[random_site].split("_"):
                    #site_index = int(site)
                    
                    #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                    
                    transition_times[int(site)][0] += t - transition_times[int(site)][2]
                    transition_times[int(site)][2] = t
                    transition_times[int(site)][3] = False

                    #df = swap_column_vals_after_k(df, site_index, df_col)
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
                
                deltaT = 1/(Q*(N_active-1))
                #t += deltaT
                #t5=time.time_ns()
                #print("heal")
                
                    
                    #t5=time.time_ns()
                    #print("inf")
                    
            else:
                #print("wasting")
                deltaT = 1/(Q*N_active)
            
                
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

def sparsified_DCP_fast_memsafe(G, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100, return_density_set = False):

    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    infection_progenitor = G.getNodeAttribute("inf_prog", int)
    nodes = list(G.iterNodes())
    
    N = G.numberOfNodes()
    
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    transition_times = {site : [0, 0, 0, False] for site in range(original_graph_size)}
    #          t_active, t_inactive, t_lastTransition, False=last transition was active->inactive
    
    
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
        
        #larger node indices appear after sparsification (even though there are fewer nodes), so this prevents out of bounds errors
        c = components[site].split("_")
        
        for component in c:
            transition_times[int(component)] = [0, 0, 0, True]
    
    step_counter = 0
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from the original sparsified_DCP_fast bc it only tracks the 
    total times that each node is infected/healthy for instead of tracking 
    every transition. 
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
    

    Q_list = []
    for u in G.iterNodes():
        sum_neighbor_weights = sum([v[1] for v in G.iterNeighborsWeights(u)])
        #a = list(G.iterNeighborsWeights(u))
        Q_i = healing_factor[u] + sum_neighbor_weights
        Q_list.append(Q_i)
    #Q_list = [healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()]
    Q = max(Q_list)#2#max([healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()])
    
    all_exposed_sites = []
    all_neighbor_edge_weights = []
    all_lambda_i = []
    
    for site in G.iterNodes():
        all_exposed_sites.append(list(G.iterNeighbors(site)))
        all_neighbor_edge_weights.append([u[1] for u in G.iterNeighborsWeights(site)])
        all_lambda_i.append(sum(all_neighbor_edge_weights[site]))
    #print(Q)
    
    #these can be any value technically, as they get multiplied by zero in the zero-th step
    density_ev = 0
    density_sq_ev = 0
    #list = np.zeros(round(t_max*Q*N)*2) #this set stores the density at every timestep for maximum of susceptibility calculations
    
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    comps = cc.getComponents()
    #component_each_node_is_in = [get_index_of_sublist_with_value(u, comps) for u in G.iterNodes()]
    biggest_component = max(comps, key=lambda x: len(x))
    #print(comps)
    
    def component_has_many_active_sites(node, biggest_component):
        #THIS TURNS OFF THE COMPONENT-WISE QUASISTATIONARY SIMULATION!!!!!
        #return True
        
        #many=2
        #nodes_of_component_this_node_is_in = comps[component_each_node_is_in[node]]
        if node in biggest_component:
            count_active  = 0
            for i in biggest_component:
                if is_active[i]:
                    count_active += 1
                if count_active > 1:
                    return True
            print("--------------------------------------------")
            print("--------------------------------------------")
            print("HIT REFLECTING BOUNDARY CONDITION!!!!!!!!!!!")
            print("--------------------------------------------")
            print("--------------------------------------------")
            return False
        else:
            return True
    
    
    while t < t_max:
        #t0=time.time_ns()
        N_active = len(active_nodes)
        
        #print(N_active)
        # the number of exposed nodes is the same as the number of edges connected to active sites
        #N_exposed = len(exposed_edges)
        #t1=time.time_ns()
        #a = [G.weight(u, v) for u,v in exposed_edges]
        #print(a)
        #print(patient_zero)
        #print(active_nodes)
        #TODO Maybe change this so that we have an absolute maximum (across whole network) that we can use? Then no need to recompute and we get rid of this expensive check
        
        if return_density_set:
            #get the current density of active sites, rho
            rho = N_active/N
            
            #update the running averages (expected values) of rho and rho squared with this method
            density_ev = ((density_ev*step_counter) + rho)/(step_counter+1)
            density_sq_ev = ((density_sq_ev*step_counter) + (rho**2))/(step_counter+1)
            """try:
                density_list[step_counter] = N_active/N
            except:
                print(f"t={t} out of {t_max} | step = {step_counter} length = {len(density_list)}")"""
            #density_set.add(N_active/N)

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        #u3 = np.random.random()
        
        
        #t1=time.time_ns()
        #t2=0
        #t3=0
        #t4=0
        #t5=0
        deltaT = 0
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = mu_i / Q #(mu_i*N_active)/((mu_i*N_active)+N_exposed)
            neighbor_edge_weights = all_neighbor_edge_weights[random_site]
            lambda_i = all_lambda_i[random_site]
            prob_to_infect = lambda_i / Q

            #t2=time.time_ns()
            #t4=time.time_ns()
            
            
            if u2 < prob_to_infect:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                exposed_sites = all_exposed_sites[random_site]
                if len(exposed_sites) > 0:
                    #exposed_sites = list(G.iterNeighbors(random_site))#[site for site in G.iterNeighbors(random_site)]
                    #temp = [u[1] for u in G.iterNeighborsWeights(random_site)]
                    exposed_sites_pmf = np.array(neighbor_edge_weights)*(1/lambda_i)
                    site_to_infect = np.random.choice(exposed_sites, p=exposed_sites_pmf)
                    
                    if not is_active[site_to_infect]:
                        #print("infecting")
                        is_active[site_to_infect] = 1
                        #print(type(random_site))
                        infection_progenitor[site_to_infect] = int(random_site)
                        
                        active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                        """newly_exposed_neighbors = [(site_to_infect,v) 
                                for v in G.iterNeighbors(site_to_infect) #inner
                            ]
                        for edge in newly_exposed_neighbors:
                            exposed_edges.add(edge)"""
                        
                        #for new_neighbor in all_exposed_sites[site_to_infect]:
                            #exposed_edges.add((site_to_infect, new_neighbor))
                            
                            #newWeight = G.weight(edge[0], edge[1])
                            #if newWeight > lambda_max:
                                #lambda_max = newWeight
                        if track_TOA:
                            TOA_tracker.add((t, site_to_infect, components[site_to_infect]))
                        
                        #df_col = find_lt(seconds, t)
                        for site in components[site_to_infect].split("_"):
                            #site_index = int(site)
                            #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                            
                            transition_times[int(site)][1] += t - transition_times[int(site)][2]
                            transition_times[int(site)][2] = t
                            transition_times[int(site)][3] = True
                            #df = swap_column_vals_after_k(df, site_index, df_col)
                        
                    #N_active += 1
                    deltaT = 1/(Q*(N_active+1))
                    #t += deltaT
            
            elif u1 <= prob_to_heal and not (quasistationary and N_active == 1) and component_has_many_active_sites(random_site, biggest_component):
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)

                is_active[random_site] = 0
                #infection_progenitor[random_site] = -1
                #newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                #for neighbor in G.iterNeighbors(random_site):
                #    if G.weight(random_site, neighbor) == lambda_max:
                #        exposed_edges.remove((random_site, neighbor))
                #        if len(exposed_edges) > 0:
                #            lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
                #        else:
                #            lambda_max = 0
                #            print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
                #            return transition_times
                #    else:
                #        exposed_edges.remove((random_site, neighbor))
                
                #for neighbor in all_exposed_sites[random_site]:
                    #exposed_edges.remove((random_site, neighbor))
                    
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                    
                #we could calculate df_col above "if N_active > 0" bc its global, but there are some wasted actions so may as well wait until we know we need it!
                #df_col = find_lt(seconds, t)
                for site in components[random_site].split("_"):
                    #site_index = int(site)
                    
                    #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                    
                    transition_times[int(site)][0] += t - transition_times[int(site)][2]
                    transition_times[int(site)][2] = t
                    transition_times[int(site)][3] = False

                    #df = swap_column_vals_after_k(df, site_index, df_col)
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
                #N_active -= 1
                deltaT = 1/(Q*(N_active-1))
                #t += deltaT
                #t5=time.time_ns()
                #print("heal")
                
                    
                    #t5=time.time_ns()
                    #print("inf")
                    
            else:
                #print("wasting")
                deltaT = 1/(Q*N_active)
            
                
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
            
    #at the end, do an instantaneous double state switch 
    for full_site in G.iterNodes():
        for site in components[full_site].split("_"):
            #site_index = int(site)
            #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
            if is_active[int(site)]:
                
                transition_times[int(site)][0] += t - transition_times[int(site)][2]
                transition_times[int(site)][2] = t
                transition_times[int(site)][3] = False
                
                transition_times[int(site)][1] += t - transition_times[int(site)][2]
                transition_times[int(site)][2] = t
                transition_times[int(site)][3] = True
                
                
            else:
                transition_times[int(site)][1] += t - transition_times[int(site)][2]
                transition_times[int(site)][2] = t
                transition_times[int(site)][3] = True
                
                transition_times[int(site)][0] += t - transition_times[int(site)][2]
                transition_times[int(site)][2] = t
                transition_times[int(site)][3] = False
    
        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker
    elif return_density_set:
        #density_list = np.trim_zeros(density_list)
        return transition_times, (density_ev, density_sq_ev)
    else:
        return transition_times


def sparsified_DCP_fast_notracking(G, track_TOA=False, quasistationary = False, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100):

    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
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
        

    #step_counter = 0 #used just to draw updates every once in a while    
        
    """
    This differs from the original sparsified_DCP_fast bc it only tracks the 
    total times that each node is infected/healthy for instead of tracking 
    every transition. 
    """
    t = 0
    #print(f"beginning infection")
    
    exposed_edges = set([
            (u,v) 
            for u in active_nodes #outer
            for v in G.iterNeighbors(u) #inner
        ])
    
    deltaT = 0    

    Q_list = []
    for u in G.iterNodes():
        sum_neighbor_weights = sum([v[1] for v in G.iterNeighborsWeights(u)])
        #a = list(G.iterNeighborsWeights(u))
        Q_i = healing_factor[u] + sum_neighbor_weights
        Q_list.append(Q_i)
    #Q_list = [healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()]
    Q = max(Q_list)#2#max([healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()])
    

    while t < t_max:        
        N_active = len(active_nodes)

        u1 = np.random.random()
        u2 = np.random.random()
        
        deltaT = 0
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = mu_i / Q #(mu_i*N_active)/((mu_i*N_active)+N_exposed)
            neighbor_edge_weights = [u[1] for u in G.iterNeighborsWeights(random_site)]
            lambda_i = sum(neighbor_edge_weights)
            prob_to_infect = lambda_i / Q

            if u2 < prob_to_infect:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                exposed_sites = list(G.iterNeighbors(random_site))
                if len(exposed_sites) > 0:
                    #exposed_sites = list(G.iterNeighbors(random_site))#[site for site in G.iterNeighbors(random_site)]
                    #temp = [u[1] for u in G.iterNeighborsWeights(random_site)]
                    exposed_sites_pmf = np.array(neighbor_edge_weights)*(1/lambda_i)
                    site_to_infect = np.random.choice(exposed_sites, p=exposed_sites_pmf)
                    
                    if not is_active[site_to_infect]:
                        #print("infecting")
                        is_active[site_to_infect] = 1
                        
                        active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)
                        newly_exposed_neighbors = [(site_to_infect,v) 
                                for v in G.iterNeighbors(site_to_infect) #inner
                            ]
                        for edge in newly_exposed_neighbors:
                            exposed_edges.add(edge)
     
                    deltaT = 1/(Q*(N_active+1))
                    #t += deltaT
            
            elif u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                
                if N_active == 1:
                    return False
                else:
                    is_active[random_site] = 0
                    
                    for neighbor in G.iterNeighbors(random_site):
                        exposed_edges.remove((random_site, neighbor))
                        
                    active_nodes.remove(random_site)
                    
                    
                    deltaT = 1/(Q*(N_active-1))
                
            else:
                deltaT = 1/(Q*N_active)
            
                
            t += deltaT
            
            #step_counter+=1
        else:
            #print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}, the last timestep was {deltaT}")
            return False
    
    return True



def sparsified_DCP_fast_memsafe_constmu(G, mu = 1, track_TOA=False, quasistationary = True, t_max = 1, random_start = True, start_node = 0, original_graph_size = 100):

    patient_zero = 0
    healing_factor = G.getNodeAttribute("mu", float)
    is_active = G.getNodeAttribute("active", int)
    components = G.getNodeAttribute("components", str)
    nodes = list(G.iterNodes())
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    transition_times = {site : [0, 0, 0, False] for site in range(original_graph_size)}
    #          t_active, t_inactive, t_lastTransition, False=last transition was active->inactive

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
        
        #larger node indices appear after sparsification (even though there are fewer nodes), so this prevents out of bounds errors
        c = components[site].split("_")
        
        for component in c:
            transition_times[int(component)] = [0, 0, 0, True]
    
    step_counter = 0 #used just to draw updates every once in a while
    TOA_tracker = set()
    
    if track_TOA:
        TOA_tracker.add((0, patient_zero, components[patient_zero]))
        
    """
    This differs from the original sparsified_DCP_fast bc it only tracks the 
    total times that each node is infected/healthy for instead of tracking 
    every transition. 
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
    

    """Q_list = []
    for u in G.iterNodes():
        sum_neighbor_weights = sum([v[1] for v in G.iterNeighborsWeights(u)])
        #a = list(G.iterNeighborsWeights(u))
        Q_i = healing_factor[u] + sum_neighbor_weights
        Q_list.append(Q_i)
    #Q_list = [healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()]
    Q = max(Q_list)#2#max([healing_factor[u] + sum([G.weight(u,v) for v in G.iterInNeighbors(u)]) for u in G.iterNodes()])
    print(Q)"""

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
        

        #print(lambda_max)        

        u1 = np.random.random()
        u2 = np.random.random()
        #u3 = np.random.random()
        
        
        #t1=time.time_ns()
        #t2=0
        #t3=0
        #t4=0
        #t5=0
        deltaT = 0
        
        if N_active > 0:
            #instead of using the standard method of converting set to list, then choosing from the list, we use this faster function. 
            #we need to choose from 0->2*original_size because after full sparsification we could get indices up to 2x the original max index
            random_site = fast_random_choose(active_nodes, 0, 2*original_graph_size)
            #random_site = np.random.choice(list(active_nodes)) #this is the only time we need active_nodes to be ordered, so we make a list for it
            mu_i = healing_factor[random_site]
            
            prob_to_heal = (N_active*mu) / ((N_active*mu) + N_exposed) #(mu_i*N_active)/((mu_i*N_active)+N_exposed)
            neighbor_edge_weights = [u[1] for u in G.iterNeighborsWeights(random_site)]
            lambda_i = sum(neighbor_edge_weights)
            prob_to_infect = N_exposed / ((N_active*mu) + N_exposed)

            #t2=time.time_ns()
            #t4=time.time_ns()
            
            
            if u2 < prob_to_infect:
                #print("spreading")
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                #transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                #exposed_sites = exposed_edges[:, 1]
                exposed_sites = list(G.iterNeighbors(random_site))
                if len(exposed_sites) > 0:
                    #exposed_sites = list(G.iterNeighbors(random_site))#[site for site in G.iterNeighbors(random_site)]
                    #temp = [u[1] for u in G.iterNeighborsWeights(random_site)]
                    exposed_sites_pmf = np.array(neighbor_edge_weights)*(1/lambda_i)
                    site_to_infect = np.random.choice(exposed_sites, p=exposed_sites_pmf)
                    
                    if not is_active[site_to_infect]:
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
                            
                            transition_times[int(site)][1] += t - transition_times[int(site)][2]
                            transition_times[int(site)][2] = t
                            transition_times[int(site)][3] = True
                            #df = swap_column_vals_after_k(df, site_index, df_col)
                        
                        
                    deltaT = 1/(N_active+1)
                    #t += deltaT
            
            elif u1 <= prob_to_heal and not (quasistationary and N_active == 1):
                #print("healing")
                #choose random node to heal?
                #site_to_heal = np.random.choice(tuple(active_nodes)) #active_nodes[np.random.randint(0, len(active_nodes))] #np.random.choice(active_nodes_list, p=heal_pmf)

                is_active[random_site] = 0
                
                #newly_unexposed_neighbors = [edge for edge in exposed_edges if edge[0] == random_site]
                #for neighbor in G.iterNeighbors(random_site):
                #    if G.weight(random_site, neighbor) == lambda_max:
                #        exposed_edges.remove((random_site, neighbor))
                #        if len(exposed_edges) > 0:
                #            lambda_max = max([G.weight(u, v) for u,v in exposed_edges]) 
                #        else:
                #            lambda_max = 0
                #            print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
                #            return transition_times
                #    else:
                #        exposed_edges.remove((random_site, neighbor))
                
                for neighbor in G.iterNeighbors(random_site):
                    exposed_edges.remove((random_site, neighbor))
                    
                active_nodes.remove(random_site)
                if track_TOA:
                    TOA_tracker.add((t, random_site, components[random_site]))
                    
                #we could calculate df_col above "if N_active > 0" bc its global, but there are some wasted actions so may as well wait until we know we need it!
                #df_col = find_lt(seconds, t)
                for site in components[random_site].split("_"):
                    #site_index = int(site)
                    
                    #df.iloc[site_index, df_col:] = np.where(df.iloc[site_index, df_col:] == True, False, True)
                    
                    transition_times[int(site)][0] += t - transition_times[int(site)][2]
                    transition_times[int(site)][2] = t
                    transition_times[int(site)][3] = False

                    #df = swap_column_vals_after_k(df, site_index, df_col)
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
                
                deltaT = 1/((N_active-1))
                #t += deltaT
                #t5=time.time_ns()
                #print("heal")
                
                    
                    #t5=time.time_ns()
                    #print("inf")
                    
            else:
                #print("wasting")
                deltaT = 1/(N_active)
            
                
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
    while N_active < G.numberOfNodes()*percent_infected and t < t_max:
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


def fast_sparsified_asymptotic_quasistationary_activity_probability(G, t_relax = 30, G_structure="chain", original_graph_size=100, dimensions = [1, 1], viz=True, title=""):
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
            plt.figure()
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
            
            plt.figure()
            
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
            fig.suptitle(title, fontsize=12)
            plt.show()

    return data


def fast_quasistationary_activity_probability_checkpoints(G, t_relax = 30, G_structure="chain", original_graph_size=100, dimensions = [1, 1], viz=True, title=""):
    relaxation_time = t_relax
    sparsified_DCP_fast(G, t_max=relaxation_time, original_graph_size=original_graph_size)
    #data is a dictionary with keys = sites and values = list of transition times
    data0 = sparsified_DCP_fast(G, t_max=relaxation_time/2, original_graph_size=original_graph_size)
    data1 = sparsified_DCP_fast(G, t_max=relaxation_time/2, original_graph_size=original_graph_size)
    
    if viz:
        #matrix = np.matrix(data.to_numpy())
        data = data0
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
            plt.figure()
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
            
            plt.figure()
            
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
            fig.suptitle(title, fontsize=12)
            plt.show()
            
        data = data1
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
            plt.figure()
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
            
            plt.figure()
            
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
            fig.suptitle(title, fontsize=12)
            plt.show()

    return data0, data1



def fast_dcp_until_quasistationary(G, init_time = 2**6, t_max=10000000, G_structure="chain", original_graph_size=100, dimensions = [1, 1], viz=True, title=""):
    #init time should be a power of 2
    # it doesn't actually have to be but it should be just b/c that makes nice and intuitive doubling numbers
    # t_max should be really really big; on order of 10 million for a cycle of length L=32
    print("starting")
    reset_to_distributed_infection(G, 1)
    #d_init = sparsified_DCP_fast(G, t_max=1, original_graph_size=original_graph_size)
    #vis_lat_advanced(d_init, original_graph_size, 1, G_structure, dimensions, f"initial state for " + title)
    #run relaxation at the start 
    sparsified_DCP_fast(G, t_max=init_time, original_graph_size=original_graph_size)
    #data0 = sparsified_DCP_fast(G, t_max=init_time, original_graph_size=original_graph_size)
    t0=time.time_ns()
    in_quasistationary = False
    t_i = init_time
    data_first_half = sparsified_DCP_fast(G, t_max=t_i, original_graph_size=original_graph_size)
    while not in_quasistationary:
        #data0 = sparsified_DCP_fast(G, t_max=t_i, original_graph_size=original_graph_size)
        data_second_half = sparsified_DCP_fast(G, t_max=t_i, original_graph_size=original_graph_size)
        
        spearman_compare_pval, lin_reg_r_squared, spearman_rho = spearman_compare(data_first_half, data_second_half, original_graph_size, t_i)
        
        if spearman_compare_pval < 0.05 and spearman_rho > 0.75:# and lin_reg_r_squared > 0.9:
            print("found quasistationary state")
            vis_lat_advanced(data_first_half, original_graph_size, t_i, G_structure, dimensions, title)
            vis_lat_advanced(data_second_half, original_graph_size, t_i, G_structure, dimensions, title)
            
            in_quasistationary = True
            return data_second_half
        else:
            t = time.time_ns()
            print(f"Quasistationary state not yet reached with p = {spearman_compare_pval}. Doubling to t_i = {2*t_i}")
            vis_lat_advanced(data_first_half, original_graph_size, t_i, G_structure, dimensions, title + f", t_i={t_i}, 1st half")
            vis_lat_advanced(data_second_half, original_graph_size, t_i, G_structure, dimensions, title + f", t_i={t_i}, 2nd half")
            time_so_far = (t-t0)/1000000000 # in secconds
            next_t_hrs = time_so_far // 3600
            next_t_mins = (time_so_far-(next_t_hrs*3600)) // 60
            next_t_secs = (time_so_far - (next_t_hrs*3600) - (next_t_mins*60)) 
            
            print(f"Simulation has taken taken {time_so_far} seconds; next will end in roughly {next_t_hrs}hrs : {next_t_mins}m : {next_t_secs}s")
            data_first_half = collate_fast_dcp_data(data_first_half, data_second_half, t_i)
            print("data collated")
            print(f"time is {time.localtime().tm_hour} : {time.localtime().tm_min} : {time.localtime().tm_sec} --- next step finishes at {time.localtime().tm_hour + next_t_hrs} : {time.localtime().tm_min + next_t_mins} : {time.localtime().tm_sec + next_t_secs}")
            t_i *= 2
        print(" ------- ")


def fast_dcp_until_quasistationary_memsafe(G, init_time = 2**3, t_max=4096, G_structure="chain", original_graph_size=100, dimensions = [1, 1], viz=True, title="", spearman_thresh = 0.99, save_last_state=False, return_density_set=False, return_time = False, print_stats = True):
    #init time should be a power of 2
    # it doesn't actually have to be but it should be just b/c that makes nice and intuitive doubling numbers
    # t_max should be really really big; on order of 10 million for a cycle of length L=32
    
    print("starting")
    reset_to_distributed_infection(G, 1)
    #d_init = sparsified_DCP_fast(G, t_max=1, original_graph_size=original_graph_size)
    #vis_lat_advanced(d_init, original_graph_size, 1, G_structure, dimensions, f"initial state for " + title)
    #run relaxation at the start 
    sparsified_DCP_fast_memsafe(G, t_max=init_time, original_graph_size=original_graph_size)
    #data0 = sparsified_DCP_fast(G, t_max=init_time, original_graph_size=original_graph_size)
    t0=time.time_ns()
    in_quasistationary = False
    t_i = init_time
    data_first_half = sparsified_DCP_fast_memsafe(G, t_max=t_i, original_graph_size=original_graph_size)
    while not in_quasistationary:
        #data0 = sparsified_DCP_fast(G, t_max=t_i, original_graph_size=original_graph_size)
        if return_density_set:
            data_second_half, density_tuples = sparsified_DCP_fast_memsafe(G, t_max=t_i, original_graph_size=original_graph_size, return_density_set=return_density_set)
        else:
            data_second_half = sparsified_DCP_fast_memsafe(G, t_max=t_i, original_graph_size=original_graph_size, return_density_set=return_density_set)
        spearman_compare_pval, lin_reg_r_squared, spearman_rho = spearman_compare_memsafe(data_first_half, data_second_half, original_graph_size, t_i, title=title + f", t_i={t_i} scatter plot", verbose = print_stats, make_fig=viz)
        
        if (spearman_compare_pval < 0.05 and spearman_rho > spearman_thresh) or t_i > t_max:# and lin_reg_r_squared > 0.9:
            if (spearman_compare_pval < 0.05 and spearman_rho > spearman_thresh):
                print("found quasistationary state")
            else:
                print("ran into t_max")
            if viz:
                vis_lat_advanced_memsafe(data_first_half, original_graph_size, t_i, G_structure, dimensions, title + f", Quasi: t_i={t_i}, 1st half")
                vis_lat_advanced_memsafe(data_second_half, original_graph_size, t_i, G_structure, dimensions, title + f", Quasi: t_i={t_i}, 2nd half")
            
            infection_progenitors = G.getNodeAttribute("inf_prog", int)
            
            infection_progenitors_list = [[] for site in G.iterNodes()]
            
            for site in G.iterNodes():
                infection_progenitors_list[infection_progenitors[site]].append(site)
            
            L = int(math.sqrt(G.numberOfNodes()))
            print(infection_progenitors_list)
            vis_given_clusters(infection_progenitors_list, L, "lattice", "Infection Progenitors Map")
            
            if save_last_state:
                vis_lat_advanced_memsafe(data_second_half, original_graph_size, t_i, G_structure, dimensions, title + f", Quasi: t_i={t_i}, 2nd half", save_fig=True)
            in_quasistationary = True
            if return_time:
                return data_second_half, t_i
            elif not return_density_set:
                return data_second_half
            
            else: 
                return data_second_half, density_tuples
        else:
            t = time.time_ns()
            print(f"Quasistationary state not yet reached with rho = {spearman_rho}. Doubling to t_i = {2*t_i}. ")
            if viz:
                vis_lat_advanced_memsafe(data_first_half, original_graph_size, t_i, G_structure, dimensions, title + f", t_i={t_i}, 1st half")
                vis_lat_advanced_memsafe(data_second_half, original_graph_size, t_i, G_structure, dimensions, title + f", t_i={t_i}, 2nd half")
            time_so_far = (t-t0)/1000000000 # in secconds
            next_t_hrs = time_so_far // 3600
            next_t_mins = (time_so_far-(next_t_hrs*3600)) // 60
            next_t_secs = (time_so_far - (next_t_hrs*3600) - (next_t_mins*60)) 
            
            print(f"Simulation has taken {time_so_far} seconds; next will end in roughly {next_t_hrs}hrs : {next_t_mins}m : {next_t_secs}s. Time is {time.localtime().tm_hour} : {time.localtime().tm_min} : {time.localtime().tm_sec} --- next step done at {time.localtime().tm_hour + next_t_hrs} : {time.localtime().tm_min + next_t_mins} : {time.localtime().tm_sec + next_t_secs}")
            data_first_half = collate_fast_dcp_memsafe_data(data_first_half, data_second_half)
            #print("data collated")
            #print(f"time is {time.localtime().tm_hour} : {time.localtime().tm_min} : {time.localtime().tm_sec} --- next step finishes at {time.localtime().tm_hour + next_t_hrs} : {time.localtime().tm_min + next_t_mins} : {time.localtime().tm_sec + next_t_secs}")
            t_i *= 2
        print(" ------- ")

def compare_color_similarity(coloring0, coloring1):
    """
    This function measures the *overlap* in two colorings of the same graph
    No "similarity score" is given for nearly-overlapping regions
    coloring0: a graph coloring, given by a list of lists
    coloring1: like coloring0

    """
    print("DOING SOMETHING")
    #coloring0 = set()
    #coloring1 = set()
    #for cluster in coloring0_list:
    #    coloring0.add(tuple(cluster))
    #for cluster in coloring1_list:
    #    coloring1.add(tuple(cluster))
    
    num_colors0 = 0
    num_colors1 = 0
    for cluster in coloring0:
        if len(cluster) > 0:
            num_colors0 += 1
            
    for cluster in coloring1:
        if len(cluster) > 0:
            num_colors1 += 1
            
    if num_colors0 > num_colors1:
        return num_colors1/num_colors0
    return num_colors0/num_colors1
    """
    similarity_counter = 0
    #for every cluster in coloring0...
    for cluster in coloring0:
        other_cluster_comparison_scores = []
        #check how similar it is to all of the clusters in coloring1
        for other_cluster in coloring1:
            cluster_cluster_similarity = 0
            for element in cluster:
                if element in other_cluster:
                    cluster_cluster_similarity += 1
                    
            other_cluster_comparison_scores.append(cluster_cluster_similarity)
        #the cluster it is most similar to will hopefully overlap it a lot
        #so we add to our similarity counter!
        #the sum of all max similarities can never be larger than the size of the graph
        similarity_with_closest_cluster = max(other_cluster_comparison_scores)
        similarity_counter += similarity_with_closest_cluster
    

    #gives 1 if they are identical, 0 if totally different
    return similarity_counter/len(coloring0)"""
        
    

def fast_dcp_until_color_quasistationary_memsafe(G, init_time = 2**6, t_max=114096, G_structure="chain", original_graph_size=100, dimensions = [1, 1], viz=True, title="", spearman_thresh = 0.99, save_last_state=False, return_density_set=False, return_time = False, print_stats = True):
    #init time should be a power of 2
    # it doesn't actually have to be but it should be just b/c that makes nice and intuitive doubling numbers
    # t_max should be really really big; on order of 10 million for a cycle of length L=32
    infection_progenitors = G.getNodeAttribute("inf_prog", int)
    print("starting")
    reset_to_distributed_infection(G, 1)
    #d_init = sparsified_DCP_fast(G, t_max=1, original_graph_size=original_graph_size)
    #vis_lat_advanced(d_init, original_graph_size, 1, G_structure, dimensions, f"initial state for " + title)
    #run relaxation at the start 
    sparsified_DCP_fast_memsafe(G, t_max=init_time, original_graph_size=original_graph_size)
    #data0 = sparsified_DCP_fast(G, t_max=init_time, original_graph_size=original_graph_size)
    old_infection_progenitors_list = [[] for site in G.iterNodes()]
    
    for site in G.iterNodes():
        old_infection_progenitors_list[infection_progenitors[site]].append(site)
    
    t0=time.time_ns()
    in_quasistationary = False
    t_i = init_time
    data_first_half = sparsified_DCP_fast_memsafe(G, t_max=t_i, original_graph_size=original_graph_size)
    while not in_quasistationary:
        #data0 = sparsified_DCP_fast(G, t_max=t_i, original_graph_size=original_graph_size)
        #run the quasistationary simulation
        if return_density_set:
            data_second_half, density_tuples = sparsified_DCP_fast_memsafe(G, t_max=t_i, original_graph_size=original_graph_size, return_density_set=return_density_set)
        else:
            data_second_half = sparsified_DCP_fast_memsafe(G, t_max=t_i, original_graph_size=original_graph_size, return_density_set=return_density_set)
        spearman_compare_pval, lin_reg_r_squared, spearman_rho = spearman_compare_memsafe(data_first_half, data_second_half, original_graph_size, t_i, title=title + f", t_i={t_i} scatter plot", verbose = print_stats, make_fig=viz)
        
        new_infection_progenitors_list = [[] for site in G.iterNodes()]
        
        for site in G.iterNodes():
            new_infection_progenitors_list[infection_progenitors[site]].append(site)
        
        color_sim = compare_color_similarity(old_infection_progenitors_list, new_infection_progenitors_list)
        L = int(math.sqrt(G.numberOfNodes()))
        vis_given_clusters(old_infection_progenitors_list, L, "lattice", "1st Infection Progenitors Map", ignore_singletons=True)

        vis_given_clusters(new_infection_progenitors_list, L, "lattice", "2nd Infection Progenitors Map", ignore_singletons=True)

        
        old_infection_progenitors_list = new_infection_progenitors_list
        #print("------")
        print(f"color similarity is {color_sim}")
        
        energy_clusters = get_energy_clusters(G, verbose=False)
        vis_given_clusters(energy_clusters, L, "lattice", "energy clusters")
        
        if color_sim > 0.999 or t_i > t_max:#(spearman_compare_pval < 0.05 and spearman_rho > spearman_thresh) or t_i > t_max:# and lin_reg_r_squared > 0.9:
            if (spearman_compare_pval < 0.05 and spearman_rho > spearman_thresh):
                print("found quasistationary state")
            else:
                print("ran into t_max")
            if viz:
                vis_lat_advanced_memsafe(data_first_half, original_graph_size, t_i, G_structure, dimensions, title + f", Quasi: t_i={t_i}, 1st half")
                vis_lat_advanced_memsafe(data_second_half, original_graph_size, t_i, G_structure, dimensions, title + f", Quasi: t_i={t_i}, 2nd half")
            
            
            
            infection_progenitors_list = [[] for site in G.iterNodes()]
            
            for site in G.iterNodes():
                infection_progenitors_list[infection_progenitors[site]].append(site)
            
            L = int(math.sqrt(G.numberOfNodes()))
            print(old_infection_progenitors_list)
            print("--")
            print(infection_progenitors_list)
            vis_given_clusters(old_infection_progenitors_list, L, "lattice", "1st half Infection Progenitors Map")

            vis_given_clusters(infection_progenitors_list, L, "lattice", "2nd half Infection Progenitors Map")
            
            if save_last_state:
                vis_lat_advanced_memsafe(data_second_half, original_graph_size, t_i, G_structure, dimensions, title + f", Quasi: t_i={t_i}, 2nd half", save_fig=True)
            in_quasistationary = True
            if return_time:
                return data_second_half, t_i
            elif not return_density_set:
                return data_second_half
            
            else: 
                return data_second_half, density_tuples
        else:
            t = time.time_ns()
            print(f"Quasistationary state not yet reached with rho = {spearman_rho}. Doubling to t_i = {2*t_i}. ")
            if viz:
                vis_lat_advanced_memsafe(data_first_half, original_graph_size, t_i, G_structure, dimensions, title + f", t_i={t_i}, 1st half")
                vis_lat_advanced_memsafe(data_second_half, original_graph_size, t_i, G_structure, dimensions, title + f", t_i={t_i}, 2nd half")
            
            
            time_so_far = (t-t0)/1000000000 # in secconds
            next_t_hrs = time_so_far // 3600
            next_t_mins = (time_so_far-(next_t_hrs*3600)) // 60
            next_t_secs = (time_so_far - (next_t_hrs*3600) - (next_t_mins*60)) 
            
            print(f"Simulation has taken {time_so_far} seconds; next will end in roughly {next_t_hrs}hrs : {next_t_mins}m : {next_t_secs}s. Time is {time.localtime().tm_hour} : {time.localtime().tm_min} : {time.localtime().tm_sec} --- next step done at {time.localtime().tm_hour + next_t_hrs} : {time.localtime().tm_min + next_t_mins} : {time.localtime().tm_sec + next_t_secs}")
            data_first_half = collate_fast_dcp_memsafe_data(data_first_half, data_second_half)
            #print("data collated")
            #print(f"time is {time.localtime().tm_hour} : {time.localtime().tm_min} : {time.localtime().tm_sec} --- next step finishes at {time.localtime().tm_hour + next_t_hrs} : {time.localtime().tm_min + next_t_mins} : {time.localtime().tm_sec + next_t_secs}")
            t_i *= 2
        print(" ------- ")



def fast_memsafe_quasistationary_density(G, init_time = 2**6, t_max=10000000, G_structure="lattice", original_graph_size=100, dimensions = [1, 1], viz=False, title="", density_thresh = 0.01):
    #init time should be a power of 2
    # it doesn't actually have to be but it should be just b/c that makes nice and intuitive doubling numbers
    # t_max should be really really big; on order of 10 million for a cycle of length L=32
    print("starting")
    reset_to_distributed_infection(G, 1)
    #d_init = sparsified_DCP_fast(G, t_max=1, original_graph_size=original_graph_size)
    #vis_lat_advanced(d_init, original_graph_size, 1, G_structure, dimensions, f"initial state for " + title)
    #run relaxation at the start 
    sparsified_DCP_fast_memsafe(G, t_max=init_time, original_graph_size=original_graph_size)
    #data0 = sparsified_DCP_fast(G, t_max=init_time, original_graph_size=original_graph_size)
    t0=time.time_ns()
    in_quasistationary = False
    t_i = init_time
    data_first_half = sparsified_DCP_fast_memsafe(G, t_max=t_i, original_graph_size=original_graph_size)
    while not in_quasistationary:
        #data0 = sparsified_DCP_fast(G, t_max=t_i, original_graph_size=original_graph_size)
        data_second_half = sparsified_DCP_fast_memsafe(G, t_max=t_i, original_graph_size=original_graph_size)
        
        first_half_density, second_half_density = density_compare_memsafe(data_first_half, data_second_half)
        if abs(first_half_density - second_half_density) < density_thresh:# and lin_reg_r_squared > 0.9:
            print("found quasistationary state")
            if viz:
                vis_lat_advanced_memsafe(data_first_half, original_graph_size, t_i, G_structure, dimensions, title)
                vis_lat_advanced_memsafe(data_second_half, original_graph_size, t_i, G_structure, dimensions, title)
            
            in_quasistationary = True
            return second_half_density
        else:
            t = time.time_ns()
            print(f"Quasistationary state not yet reached with delta density = {abs(first_half_density - second_half_density)}. Doubling to t_i = {2*t_i}")
            if viz:
                vis_lat_advanced_memsafe(data_first_half, original_graph_size, t_i, G_structure, dimensions, title + f", t_i={t_i}, 1st half")
                vis_lat_advanced_memsafe(data_second_half, original_graph_size, t_i, G_structure, dimensions, title + f", t_i={t_i}, 2nd half")
            time_so_far = (t-t0)/1000000000 # in secconds
            next_t_hrs = time_so_far // 3600
            next_t_mins = (time_so_far-(next_t_hrs*3600)) // 60
            next_t_secs = (time_so_far - (next_t_hrs*3600) - (next_t_mins*60)) 
            
            print(f"Simulation has taken taken {time_so_far} seconds; next will end in roughly {next_t_hrs}hrs : {next_t_mins}m : {next_t_secs}s")
            data_first_half = collate_fast_dcp_memsafe_data(data_first_half, data_second_half)
            print("data collated")
            print(f"time is {time.localtime().tm_hour} : {time.localtime().tm_min} : {time.localtime().tm_sec} --- next step finishes at {time.localtime().tm_hour + next_t_hrs} : {time.localtime().tm_min + next_t_mins} : {time.localtime().tm_sec + next_t_secs}")
            t_i *= 2
        print(" ------- ")



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
    G_sdrg = copy_graph(G)
    G_effr = copy_graph(G)
    B = copy_graph(G)
    
    B, n = SMDS_sparsifier(B, chi, output_num_components = True)
    
    G_sdrg = sdrg_sparsify(G_sdrg)
    #until_n_components(G_tilde, n)
    
    print(f"metric backbone has: {B.numberOfEdges()} edges")
    print(f"sdrg has: {G_sdrg.numberOfEdges()} edges")

    G_effr = effective_resistance_sampling_sparsification(G_effr, B.numberOfEdges())
    
    
    print("starting t_half measurements")
    print("std - ")
    t_half_G = t_half(G, sim_time, num_simulations=num_simulations, original_graph_size=G.numberOfNodes())
    print("sdrg - ")
    t_half_G_sdrg = t_half(G_sdrg, sim_time, num_simulations=num_simulations, original_graph_size=G.numberOfNodes())
    print("smds - ")
    t_half_B = t_half(B, sim_time, num_simulations=num_simulations, original_graph_size=G.numberOfNodes())
    print("effr - ")
    t_half_G_effr = t_half(G_effr, sim_time, num_simulations=num_simulations, original_graph_size=G.numberOfNodes())

    #print(f"t_1/2 SDRG / t_1/2 is {t_half_G_tilde/t_half_G}")
    #print(f"t_1/2 semi-metric / t_1/2 is {t_half_B/t_half_G}")
    print("-- vals --")
    print(f"std: {t_half_G}")
    print(f"smds: {t_half_B}")
    print(f"sdrg: {t_half_G_sdrg}")
    print(f"effr {t_half_G_effr}")
    return (t_half_G, t_half_B, t_half_G_sdrg, t_half_G_effr)
 
def generate_xi_half_curve(G, sim_time, num_simulations):
    
    chi_values = np.linspace(0,1, 5)
    
    sdrg_xi_vals = []
    smds_xi_vals = []  
    effr_xi_vals = []  
    
    x = []
    
    for chi in chi_values:
        vals = xi_half_chi(G, chi, sim_time, num_simulations=num_simulations)        
        print(vals)
        sdrg_xi_vals.append(vals[2]/vals[0])
        smds_xi_vals.append(vals[1]/vals[0])
        effr_xi_vals.append(vals[3]/vals[0])
        x.append(chi)
        
    print(sdrg_xi_vals)
    print(smds_xi_vals)
    print(effr_xi_vals)
    
    plt.plot(x, sdrg_xi_vals)
    plt.plot(x, smds_xi_vals)
    plt.plot(x, effr_xi_vals)
    plt.title("xi^x_{1/2} values for SDRG, SMDS, effR")
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
    
    proportions_of_time_active_o = total_time_active_list/relaxation_time
    prop_orig = [(proportions_of_time_active_o[i], i) for i in range(len(proportions_of_time_active_o)) ]
    
    prop_mod = []
    data = mod_data
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        trans_times = data[site]
        total_time_active = 0
        for i in range(1, len(trans_times), 2):
            total_time_active += trans_times[i] - trans_times[i-1]
        total_time_active_list[site] = total_time_active
    
    proportions_of_time_active_m = total_time_active_list/relaxation_time
    prop_mod = [(proportions_of_time_active_m[i], i) for i in range(len(proportions_of_time_active_m)) ]
    a = []
    average_activity_comparison = np.average(abs(proportions_of_time_active_o - proportions_of_time_active_m))
    prop_orig = sorted(prop_orig)
    sorted_prop_time_active_o = sorted(proportions_of_time_active_o)
    sorted_indices = [prop_orig[i][1] for i in range(len(prop_orig))]
    prop_mod_sorted = []
    for index in sorted_indices:
        prop_mod_sorted.append(prop_mod[index][0])
        a.append((prop_mod[index][0], prop_mod[index][1]))
    #prop_mod = sorted(prop_mod, key=lambda x: x[1])
    
    spearman_correlation = scipy.stats.spearmanr(sorted_prop_time_active_o, prop_mod_sorted)
    
    wasserstein_dist = scipy.stats.wasserstein_distance(sorted_prop_time_active_o, prop_mod_sorted)
    linr = scipy.stats.linregress(sorted_prop_time_active_o, prop_mod_sorted)
    print(f"{linr.slope}, {linr.intercept}, {linr.pvalue}, {linr.stderr}, {linr.intercept_stderr}")
    #print(prop_orig[0:25])
    #print(a[0:25])
    #x = np.random.rand(len(prop_mod))
    #y = np.random.rand(len(prop_mod))
    plt.figure()
    plt.scatter(sorted_prop_time_active_o, prop_mod_sorted)
    x=np.linspace(0,1)#max(max(sorted_prop_time_active_o), max(prop_mod_sorted)))
    plt.plot(x, linr.intercept + linr.slope*x, 'r', label='fitted line')
    print(f"linear regression R^2: {linr.rvalue}")
    print(f"absolute node-by-node activity difference, averaged: {average_activity_comparison}")
    print(f"spearman rho: {spearman_correlation.statistic}")
    print(f"spearman p-value: {spearman_correlation.pvalue}")
    print(f"wasserstein distance: {wasserstein_dist}")


def spearman_compare(orig_data, mod_data, original_graph_size, relaxation_time):
    #get prop_time_active_orig
    #data_arr = [orig_data, mod_data]
    #prop_times_active = np.empty(2)
    
    prop_orig = []
    data = orig_data
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        trans_times = data[site]
        total_time_active = 0
        for i in range(1, len(trans_times), 2):
            total_time_active += trans_times[i] - trans_times[i-1]
        total_time_active_list[site] = total_time_active
    
    proportions_of_time_active_o = total_time_active_list/relaxation_time
    prop_orig = [(proportions_of_time_active_o[i], i) for i in range(len(proportions_of_time_active_o)) ]
    
    prop_mod = []
    data = mod_data
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        trans_times = data[site]
        total_time_active = 0
        for i in range(1, len(trans_times), 2):
            total_time_active += trans_times[i] - trans_times[i-1]
        total_time_active_list[site] = total_time_active
    
    proportions_of_time_active_m = total_time_active_list/relaxation_time
    prop_mod = [(proportions_of_time_active_m[i], i) for i in range(len(proportions_of_time_active_m)) ]
    a = []
    average_activity_comparison = np.average(abs(proportions_of_time_active_o - proportions_of_time_active_m))
    prop_orig = sorted(prop_orig)
    sorted_prop_time_active_o = sorted(proportions_of_time_active_o)
    sorted_indices = [prop_orig[i][1] for i in range(len(prop_orig))]
    prop_mod_sorted = []
    for index in sorted_indices:
        prop_mod_sorted.append(prop_mod[index][0])
        a.append((prop_mod[index][0], prop_mod[index][1]))
    #prop_mod = sorted(prop_mod, key=lambda x: x[1])
    
    spearman_correlation = scipy.stats.spearmanr(sorted_prop_time_active_o, prop_mod_sorted)
    
    wasserstein_dist = scipy.stats.wasserstein_distance(sorted_prop_time_active_o, prop_mod_sorted)
    linr = scipy.stats.linregress(sorted_prop_time_active_o, prop_mod_sorted)
    print(f"{linr.slope}, {linr.intercept}, {linr.pvalue}, {linr.stderr}, {linr.intercept_stderr}")
    #print(prop_orig[0:25])
    #print(a[0:25])
    #x = np.random.rand(len(prop_mod))
    #y = np.random.rand(len(prop_mod))
    plt.figure()
    plt.scatter(sorted_prop_time_active_o, prop_mod_sorted)
    x=np.linspace(0,1)#max(max(sorted_prop_time_active_o), max(prop_mod_sorted)))
    plt.plot(x, linr.intercept + linr.slope*x, 'r', label='fitted line')
    print(f"linear regression R^2: {linr.rvalue**2}")
    print(f"absolute node-by-node activity difference, averaged: {average_activity_comparison}")
    print(f"spearman rho: {spearman_correlation.statistic}")
    print(f"spearman p-value: {spearman_correlation.pvalue}")
    print(f"wasserstein distance: {wasserstein_dist}")
    
    return spearman_correlation.pvalue, linr.rvalue**2, spearman_correlation.statistic



def spearman_compare_memsafe(orig_data, mod_data, original_graph_size, relaxation_time, title = "", verbose=True, save_fig = False, make_fig=True):
    #get prop_time_active_orig
    #data_arr = [orig_data, mod_data]
    #prop_times_active = np.empty(2)
    
    prop_orig = []
    data = orig_data
        
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        
        total_time_active_list[site] = data[site][0]
    
    proportions_of_time_active_o = total_time_active_list/relaxation_time
    
    #proportions_of_time_active_o = total_time_active_list/relaxation_time
    prop_orig = [(proportions_of_time_active_o[i], i) for i in range(len(proportions_of_time_active_o)) ]
    
    prop_mod = []
    data = mod_data
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        
        total_time_active_list[site] = data[site][0]
    
    proportions_of_time_active_m = total_time_active_list/relaxation_time
    prop_mod = [(proportions_of_time_active_m[i], i) for i in range(len(proportions_of_time_active_m)) ]
    #a = []
    average_activity_comparison = np.average(abs(proportions_of_time_active_o - proportions_of_time_active_m))
    prop_orig = sorted(prop_orig)
    sorted_prop_time_active_o = sorted(proportions_of_time_active_o)
    sorted_indices = [prop_orig[i][1] for i in range(len(prop_orig))]
    prop_mod_sorted = []
    for index in sorted_indices:
        prop_mod_sorted.append(prop_mod[index][0])
        #a.append((prop_mod[index][0], prop_mod[index][1]))
    #prop_mod = sorted(prop_mod, key=lambda x: x[1])
    
    spearman_correlation = scipy.stats.spearmanr(sorted_prop_time_active_o, prop_mod_sorted)
    
    wasserstein_dist = scipy.stats.wasserstein_distance(sorted_prop_time_active_o, prop_mod_sorted)
    linr = scipy.stats.linregress(sorted_prop_time_active_o, prop_mod_sorted)
    if verbose:
        print(f"{linr.slope}, {linr.intercept}, {linr.pvalue}, {linr.stderr}, {linr.intercept_stderr}")
    #print(prop_orig[0:25])
    #print(a[0:25])
    #x = np.random.rand(len(prop_mod))
    #y = np.random.rand(len(prop_mod))
    if make_fig:
        plt.figure()
        plt.scatter(sorted_prop_time_active_o, prop_mod_sorted)
        x=np.linspace(0,1)#max(max(sorted_prop_time_active_o), max(prop_mod_sorted)))
        plt.suptitle(title)
    
        plt.xlabel("1st half")
        plt.ylabel("2nd half")
        plt.plot(x, linr.intercept + linr.slope*x, 'r', label='fitted line')
        if save_fig:
            plt.savefig(f"{title}.png", dpi=600)
    if verbose:
        print(f"linear regression R^2: {linr.rvalue**2}")
        print(f"absolute node-by-node activity difference, averaged: {average_activity_comparison}")
        print(f"spearman rho: {spearman_correlation.statistic}")
        print(f"spearman p-value: {spearman_correlation.pvalue}")
        print(f"wasserstein distance: {wasserstein_dist}")
    
    return spearman_correlation.pvalue, linr.rvalue**2, spearman_correlation.statistic


def spearman_compare_memsafe_diff(orig_data, mod_data, original_graph_size, relaxation_time_orig, relaxation_time_mod, important_nodes = [], wass_out=False, jensen_out = False, title = "", save_fig=False, verbose = True, show_fig=True):
    #get prop_time_active_orig
    #data_arr = [orig_data, mod_data]
    #prop_times_active = np.empty(2)
    
    prop_orig = []
    data = orig_data
    
    #print(orig_data)
    #print(mod_data)
        
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        
        total_time_active_list[site] = data[site][0]
        #print(site)
    
    proportions_of_time_active_o = total_time_active_list/relaxation_time_orig
    for i in range(len(proportions_of_time_active_o)):
        if proportions_of_time_active_o[i] < 1e-320:
            proportions_of_time_active_o[i] = 1e-320
    #times_active_orig = proportions_of_time_active_o
    
    #proportions_of_time_active_o = total_time_active_list/relaxation_time
    prop_orig = [(proportions_of_time_active_o[i], i) for i in range(len(proportions_of_time_active_o)) ]
    
    prop_mod = []
    data = mod_data
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        
        total_time_active_list[site] = data[site][0]
        #print(site)
    proportions_of_time_active_m = total_time_active_list/relaxation_time_mod
    
    for i in range(len(proportions_of_time_active_m)):
        if proportions_of_time_active_m[i] < 1e-320:
            proportions_of_time_active_m[i] = 1e-320
    
    prop_mod = [(proportions_of_time_active_m[i], i) for i in range(len(proportions_of_time_active_m)) ]
    #a = []
    average_activity_comparison = np.average(abs(proportions_of_time_active_o - proportions_of_time_active_m))
    prop_orig = sorted(prop_orig)
    sorted_prop_time_active_o = sorted(proportions_of_time_active_o)
    sorted_indices = [prop_orig[i][1] for i in range(len(prop_orig))]
    prop_mod_sorted = []
    for index in sorted_indices:
        prop_mod_sorted.append(prop_mod[index][0])        #a.append((prop_mod[index][0], prop_mod[index][1]))
    #prop_mod = sorted(prop_mod, key=lambda x: x[1])
    
    def density_power_divergence(list1, list2, alpha=0.5, method='histogram', bins=50, 
                                bandwidth=None, return_components=False):
        """
        Calculate the density power divergence between two lists of values.
        
        Parameters:
        -----------
        list1, list2 : array-like
            Input lists of values to compare
        alpha : float, optional (default=0.5)
            The power parameter for divergence (0 <= alpha <= 1)
            - alpha=0 gives KL divergence
            - alpha=1 gives L2 distance between densities
        method : str, optional (default='histogram')
            Method for density estimation:
            - 'histogram': Use histogram-based density estimation
            - 'kde': Use kernel density estimation
        bins : int, optional (default=50)
            Number of bins for histogram method
        bandwidth : float, optional (default=None)
            Bandwidth for KDE method (if None, uses Scott's rule)
        return_components : bool, optional (default=False)
            If True, return individual divergence components
        
        Returns:
        --------
        float or tuple
            The density power divergence value, or (divergence, components) if return_components=True
        """
        
        # Convert to numpy arrays
        list1 = np.asarray(list1)
        list2 = np.asarray(list2)
        
        # Validate input
        if len(list1) == 0 or len(list2) == 0:
            raise ValueError("Input lists cannot be empty")
        
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")
        
        # Determine the range for density estimation
        min_val = min(list1.min(), list2.min())
        max_val = max(list1.max(), list2.max())
        range_val = max_val - min_val
        
        if range_val == 0:
            # If all values are the same, add some padding
            min_val -= 1
            max_val += 1
        
        # Estimate densities
        if method.lower() == 'histogram':
            # Use histogram-based density estimation
            hist1, edges = np.histogram(list1, bins=bins, range=(min_val, max_val), density=True)
            hist2, _ = np.histogram(list2, bins=bins, range=(min_val, max_val), density=True)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            hist1 = hist1 + epsilon
            hist2 = hist2 + epsilon
            
            # Normalize to ensure they sum to 1
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            p = hist1
            q = hist2
            
        elif method.lower() == 'kde':
            # Use kernel density estimation
            if bandwidth is None:
                # Scott's rule for bandwidth selection
                n1, n2 = len(list1), len(list2)
                bandwidth = 1.06 * min(np.std(list1), np.std(list2)) * min(n1, n2) ** (-1/5)
            
            # Create KDE objects
            kde1 = scipy.stats.gaussian_kde(list1, bw_method=bandwidth)
            kde2 = scipy.stats.gaussian_kde(list2, bw_method=bandwidth)
            
            # Evaluate on a grid
            grid = np.linspace(min_val, max_val, max(bins, 100))
            p = kde1(grid)
            q = kde2(grid)
            
            # Normalize
            p = p / p.sum()
            q = q / q.sum()
            
        else:
            raise ValueError("method must be 'histogram' or 'kde'")
        
        # Calculate density power divergence
        if alpha == 0:
            # KL divergence (limit case)
            # D_KL(p||q) = sum(p * log(p/q))
            divergence = np.sum(p * np.log(p / q))
            
        elif alpha == 1:
            # L2 distance between densities
            divergence = np.sum((p - q) ** 2)
            
        else:
            # General density power divergence formula:
            # D_alpha(p||q) = (1/(alpha*(alpha-1))) * sum(p^alpha * q^(1-alpha) - alpha*p + (alpha-1)*q)
            
            # Calculate the cross entropy term
            cross_entropy_term = np.sum(p ** alpha * q ** (1 - alpha))
            
            # Calculate the self entropy terms
            p_self_entropy = np.sum(p ** alpha)
            q_self_entropy = np.sum(q ** alpha)
            
            # Density power divergence
            divergence = (1 / (alpha * (alpha - 1))) * (
                cross_entropy_term - 
                (alpha * p_self_entropy + (1 - alpha) * q_self_entropy)
            )
        
        if return_components:
            components = {
                'p': p,
                'q': q,
                'density1': p,
                'density2': q
            }
            return divergence, components
        
        return divergence
        
    def least_squares_error(list1, list2, normalize=False, return_details=False):
        """
        Calculate the least squares error between two lists of values.
        
        Parameters:
        -----------
        list1, list2 : array-like
            Input lists of values to compare
        normalize : bool, optional (default=False)
            If True, normalize the error by the number of elements (MSE)
            If False, return the sum of squared errors (SSE)
        return_details : bool, optional (default=False)
            If True, return additional statistics
        
        Returns:
        --------
        float or dict
            The least squares error, or dictionary with details if return_details=True
        """
        
        # Convert to numpy arrays
        arr1 = np.asarray(list1, dtype=float)
        arr2 = np.asarray(list2, dtype=float)
        
        # Validate input
        if len(arr1) != len(arr2):
            raise ValueError(f"Lists must have the same length. Got {len(arr1)} and {len(arr2)}")
        
        if len(arr1) == 0:
            raise ValueError("Input lists cannot be empty")
        
        # Calculate squared differences
        squared_diff = (arr1 - arr2) ** 2
        
        # Calculate sum of squared errors (SSE)
        sse = np.sum(squared_diff)
        
        # Calculate mean squared error (MSE)
        mse = sse / len(arr1)
        
        # Calculate root mean squared error (RMSE)
        rmse = np.sqrt(mse)
        
        # Calculate other useful statistics
        mae = np.mean(np.abs(arr1 - arr2))  # Mean absolute error
        max_error = np.max(np.abs(arr1 - arr2))  # Maximum absolute error
        
        # R-squared (coefficient of determination)
        ss_res = sse  # Residual sum of squares
        ss_tot = np.sum((arr2 - np.mean(arr2)) ** 2)  # Total sum of squares
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0 if ss_res == 0 else -np.inf
        
        if return_details:
            return {
                'sse': sse,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error,
                'r_squared': r_squared,
                'squared_errors': squared_diff,
                'errors': arr1 - arr2
            }
        
        # Return the appropriate error metric
        if normalize:
            return mse  # Mean squared error
        else:
            return sse  # Sum of squared errors
        
    def weighted_kl_divergence(p, q, weights=None, epsilon=1e-10, normalize=True, 
                          axis=None, return_components=False):
        """
        Calculate the weighted Kullback-Leibler divergence between two distributions.
        
        D_KL(P||Q) = sum(w_i * p_i * log(p_i / q_i))
        
        Parameters:
        -----------
        p, q : array-like
            Input probability distributions (must be non-negative)
        weights : array-like, optional
            Weights for each element. If None, all weights are 1 (standard KL divergence)
        epsilon : float, optional (default=1e-10)
            Small value to avoid log(0) and division by zero
        normalize : bool, optional (default=True)
            If True, normalize p and q to sum to 1 before calculation
        axis : int, optional (default=None)
            Axis along which the distributions are defined
        return_components : bool, optional (default=False)
            If True, return individual components of the divergence
        
        Returns:
        --------
        float or tuple
            The weighted KL divergence, or (divergence, components) if return_components=True
        """
        
        # Convert to numpy arrays
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        
        # Validate input
        if p.shape != q.shape:
            raise ValueError(f"p and q must have the same shape. Got {p.shape} and {q.shape}")
        
        if np.any(p < 0) or np.any(q < 0):
            raise ValueError("Distributions must be non-negative")
        
        # Handle weights
        if weights is None:
            weights = np.ones_like(p)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != p.shape:
                raise ValueError(f"weights must have the same shape as p and q. Got {weights.shape} and {p.shape}")
            if np.any(weights < 0):
                raise ValueError("Weights must be non-negative")
        
        # Normalize if requested
        if normalize:
            p_sum = p.sum(axis=axis, keepdims=True)
            q_sum = q.sum(axis=axis, keepdims=True)
            
            # Avoid division by zero
            p_sum = np.where(p_sum > 0, p_sum, 1)
            q_sum = np.where(q_sum > 0, q_sum, 1)
            
            p = p / p_sum
            q = q / q_sum
        
        # Add epsilon to avoid numerical issues
        p = np.clip(p, epsilon, None)
        q = np.clip(q, epsilon, None)
        
        # Calculate weighted KL divergence
        # D_KL(P||Q) = sum(w_i * p_i * log(p_i / q_i))
        kl_components = weights * p * np.log(p / q)
        
        # Handle cases where p_i = 0 (0 * log(0) = 0 by convention)
        kl_components = np.where(p <= epsilon, 0, kl_components)
        
        # Sum along the specified axis
        if axis is None:
            kl_divergence = np.sum(kl_components)
        else:
            kl_divergence = np.sum(kl_components, axis=axis)
        
        if return_components:
            return kl_divergence, kl_components
        
        return kl_divergence
    
    def weighted_kl_divergence_symmetric(p, q, weights=None, method='jeffreys', **kwargs):
        """
        Calculate symmetric weighted KL divergence between two distributions.
        
        Parameters:
        -----------
        p, q : array-like
            Input probability distributions
        weights : array-like, optional
            Weights for each element
        method : str, optional (default='jeffreys')
            Method for symmetrization:
            - 'jeffreys': (KL(P||Q) + KL(Q||P)) / 2
            - 'jensen-shannon': Jensen-Shannon divergence
        **kwargs : additional arguments for weighted_kl_divergence
        
        Returns:
        --------
        float
            The symmetric weighted KL divergence
        """
        
        if method == 'jeffreys':
            kl_pq = weighted_kl_divergence(p, q, weights=weights, **kwargs)
            kl_qp = weighted_kl_divergence(q, p, weights=weights, **kwargs)
            return (kl_pq + kl_qp) / 2
        
        elif method == 'jensen-shannon':
            # Jensen-Shannon divergence
            p = np.asarray(p, dtype=float)
            q = np.asarray(q, dtype=float)
            
            # Handle weights
            if weights is None:
                weights = np.ones_like(p)
            else:
                weights = np.asarray(weights, dtype=float)
            
            # Calculate mixture distribution
            m = 0.5 * (p + q)
            
            # Weighted JS divergence
            kl_pm = weighted_kl_divergence(p, m, weights=weights, **kwargs)
            kl_qm = weighted_kl_divergence(q, m, weights=weights, **kwargs)
            
            return 0.5 * kl_pm + 0.5 * kl_qm
        
        else:
            raise ValueError("method must be 'jeffreys' or 'jensen-shannon'")
    
    def compute_weighted_wasserstein_2d(p, q, site_weights):
        """
        Computes the 2D weighted Wasserstein-1 distance between two discrete 
        probability distributions p and q, using site_weights to adjust the ground cost.
        
        Parameters:
        -----------
        p : np.ndarray (2D)
            Source probability distribution matrix.
        q : np.ndarray (2D)
            Target probability distribution matrix.
        site_weights : np.ndarray (2D)
            Importance/weight scaling factors for each site (i, j).
            
        Returns:
        --------
        float
            The weighted Wasserstein distance.
        """
        # 1. Validation and flattening
        assert p.shape == q.shape == site_weights.shape, "All arrays must share the same 2D shape."
        H, W = p.shape
        N = H * W
        
        p_flat = p.ravel() / p.sum()
        q_flat = q.ravel() / q.sum()
        w_flat = site_weights.ravel()
        
        # 2. Build the coordinate system
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        coords = np.vstack([Y.ravel(), X.ravel()]).T  # Dimensions: (N, 2)
        
        # 3. Generate the distance matrix
        # Compute Euclidean distance between every combination of coordinates
        diff = coords[:, None, :] - coords[None, :, :]  # Shape: (N, N, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)      # Shape: (N, N)
        
        # 4. Integrate site weights into the ground cost matrix
        # Cost = Euclidean distance * average importance weight of source and target sites
        weight_matrix = (w_flat[:, None] + w_flat[None, :]) / 2.0
        C = dist_matrix * weight_matrix
        c = C.ravel()  # Flatten for linear programming objective
        
        # 5. Build linear constraints (Mass conservation)
        # Row constraints: Sum of mass exiting source i must equal p[i]
        A_p = np.zeros((N, N * N))
        for i in range(N):
            A_p[i, i * N : (i + 1) * N] = 1
            
        # Column constraints: Sum of mass entering target j must equal q[j]
        A_q = np.zeros((N, N * N))
        for j in range(N):
            A_q[j, j::N] = 1
            
        # Remove final row to eliminate redundant degrees of freedom 
        A_eq = np.vstack([A_p, A_q[:-1]]) 
        b_eq = np.concatenate([p_flat, q_flat[:-1]])
        
        # 6. Solve the linear programming optimization problem
        res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
        
        if res.success:
            return res.fun
        else:
            raise ValueError(f"Optimization failed: {res.message}")
    
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    
    def wasserstein_1_distance(p, q, weights=None, metric='euclidean'):
        """
        Compute the Wasserstein-1 distance between two 2D discrete probability distributions.
        
        Parameters:
        -----------
        p : numpy.ndarray of shape (n_points, 2)
            First probability distribution as array of points with weights.
            Each row: [x, y, probability] or if p_weights provided separately.
        
        q : numpy.ndarray of shape (m_points, 2)
            Second probability distribution as array of points with weights.
            Each row: [x, y, probability] or if q_weights provided separately.
        
        weights : numpy.ndarray of shape (height, width), optional
            2D weighting array for the space. If provided, points are weighted
            by these values before computing distances.
        
        metric : str or callable, default='euclidean'
            Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
            or any metric supported by scipy.spatial.distance.cdist.
        
        Returns:
        --------
        float : Wasserstein-1 distance
        
        Examples:
        ---------
        >>> # Two simple distributions
        >>> p = np.array([[0, 0, 0.5], [1, 1, 0.5]])
        >>> q = np.array([[0, 1, 0.5], [1, 0, 0.5]])
        >>> dist = wasserstein_1_distance(p, q)
        
        >>> # With spatial weighting
        >>> weight_grid = np.ones((10, 10))
        >>> dist = wasserstein_1_distance(p, q, weights=weight_grid)
        """
        print("000000000000")
        print(len(p))
        print(len(q))
        # Separate coordinates and probabilities
        if p.shape[1] == 3:
            p_coords = p[:, :2]
            p_probs = p[:, 2]
        else:
            raise ValueError("p must have shape (n, 3) with columns [x, y, prob]")
        
        if q.shape[1] == 3:
            q_coords = q[:, :2]
            q_probs = q[:, 2]
        else:
            raise ValueError("q must have shape (m, 3) with columns [x, y, prob]")
        
        # Normalize probabilities (ensure they sum to 1)
        p_probs = p_probs / np.sum(p_probs)
        q_probs = q_probs / np.sum(q_probs)
        
        # Apply spatial weighting if provided
        if weights is not None:
            # Convert coordinates to integer indices (assuming coordinates are grid points)
            p_indices = p_coords.astype(int)
            q_indices = q_coords.astype(int)
            
            # Get weights at these positions
            p_weights = weights[p_indices[:, 0], p_indices[:, 1]]
            q_weights = weights[q_indices[:, 0], q_indices[:, 1]]
            
            # Scale probabilities by spatial weights
            p_probs = p_probs * p_weights
            q_probs = q_probs * q_weights
            
            # Renormalize
            p_probs = p_probs / np.sum(p_probs)
            q_probs = q_probs / np.sum(q_probs)
        
        # Compute cost matrix (distances between all pairs of points)
        cost_matrix = cdist(p_coords, q_coords, metric=metric)
        
        # Solve the optimal transport problem using Hungarian algorithm
        # This minimizes sum_i sum_j T_ij * cost_ij subject to:
        # sum_j T_ij = p_probs[i] and sum_i T_ij = q_probs[j]
        
        # For discrete distributions, we need to handle the case where
        # probabilities might be very small or zero
        # We'll use linear programming approach via linear_sum_assignment
        
        # Scale the cost matrix by probabilities to handle unequal masses
        # We replicate points according to their probability mass
        # This is a simple but approximate method for discrete distributions
        
        # Alternative: Use the exact linear programming formulation
        # For the exact solution, we can use scipy.optimize.linprog
        # but the Hungarian algorithm works well for equal total mass
        
        # Method 1: Using linear_sum_assignment (for equal number of points)
        if len(p_probs) == len(q_probs):
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            distance = np.sum(cost_matrix[row_ind, col_ind] * p_probs)
        else:
            print("AAAAAAAAHHHHHHHHHHH")
            # Method 2: For unequal number of points, use the exact OT solution
            # This uses the Kantorovich formulation
            from scipy.optimize import linprog
            
            n = len(p_probs)
            m = len(q_probs)
            
            # Flatten the transportation matrix variables
            # We want to minimize sum_{i,j} T_{ij} * C_{ij}
            c = cost_matrix.flatten()
            
            # Constraints:
            # For each i: sum_j T_{ij} = p_probs[i]
            # For each j: sum_i T_{ij} = q_probs[j]
            # T_{ij} >= 0
            
            # Equality constraints matrix
            A_eq = np.zeros((n + m, n * m))
            b_eq = np.concatenate([p_probs, q_probs])
            
            # Row constraints: for each i
            for i in range(n):
                A_eq[i, i*m:(i+1)*m] = 1
            
            # Column constraints: for each j
            for j in range(m):
                A_eq[n + j, j::m] = 1
            
            # Solve the linear program
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
            
            if result.success:
                distance = result.fun
            else:
                raise RuntimeError("Linear programming failed to find optimal transport")
        
        return distance
    
    
    def wasserstein_1_distance_2d_grid(p_grid, q_grid, weights=None, metric='euclidean'):
        """
        Compute Wasserstein-1 distance between two 2D distributions defined on a grid.
        
        Parameters:
        -----------
        p_grid : numpy.ndarray of shape (height, width)
            First probability distribution on a 2D grid.
        
        q_grid : numpy.ndarray of shape (height, width)
            Second probability distribution on a 2D grid.
        
        weights : numpy.ndarray of shape (height, width), optional
            2D weighting array for the space.
        
        metric : str, default='euclidean'
            Distance metric to use.
        
        Returns:
        --------
        float : Wasserstein-1 distance
        """
        # Convert grid to point list with coordinates
        height, width = p_grid.shape
        
        # Flatten and get coordinates
        p_flat = p_grid.flatten()
        q_flat = q_grid.flatten()
        
        # Get non-zero indices
        p_indices = np.where(p_flat > 0)[0]
        q_indices = np.where(q_flat > 0)[0]
        
        # Convert to coordinates
        p_rows, p_cols = np.unravel_index(p_indices, (height, width))
        q_rows, q_cols = np.unravel_index(q_indices, (height, width))
        
        # Create point arrays with probabilities
        p_points = np.column_stack([p_cols, p_rows, p_flat[p_indices]])
        q_points = np.column_stack([q_cols, q_rows, q_flat[q_indices]])
        
        print(p_points)
        print("delta")
        print(q_points)

        # Compute Wasserstein distance
        return wasserstein_1_distance(p_points, q_points, weights, metric)
    
    import ot
    
    def wasserstein_2d_grid_pot(P, Q, L=None, metric='euclidean', return_plan=False):
        """
        Compute the 2D Wasserstein-1 distance between two LxL grid probability distributions
        using POT (Python Optimal Transport) library.
        
        Parameters:
        -----------
        P : numpy.ndarray of shape (L, L)
            First probability distribution on the grid
            
        Q : numpy.ndarray of shape (L, L)
            Second probability distribution on the grid
            
        L : int, optional
            Grid size. If None, inferred from P.shape[0]
            
        metric : str, default='euclidean'
            Metric for cost matrix. Options: 'euclidean', 'manhattan', 'cosine'
            
        return_plan : bool, default=False
            If True, also return the optimal transport plan
            
        Returns:
        --------
        float : Wasserstein-1 distance
        (optional) numpy.ndarray : Optimal transport plan of shape (N, N) where N = L*L
        """
        
        # Get grid size
        if L is None:
            L = P.shape[0]
        
        # Ensure distributions sum to 1
        P = P / np.sum(P)
        Q = Q / np.sum(Q)
        #print(Q)
        
        # Flatten the distributions
        p_flat = P.flatten()
        q_flat = Q.flatten()
        
        # Remove zero-mass points for efficiency (optional but recommended)
        # Keep only points with non-zero mass
        p_nonzero = p_flat > 1e-322
        q_nonzero = q_flat > 1e-322
        
        # If we remove zeros, we need to remap indices
        if np.sum(p_nonzero) < L*L or np.sum(q_nonzero) < L*L:
            # Get coordinates of non-zero points
            p_indices = np.where(p_nonzero)[0]
            q_indices = np.where(q_nonzero)[0]
            
            p_flat = p_flat[p_indices]
            q_flat = q_flat[q_indices]
            
            # Convert flattened indices to 2D coordinates
            p_coords = np.array(np.unravel_index(p_indices, (L, L))).T
            q_coords = np.array(np.unravel_index(q_indices, (L, L))).T
        else:
            # All points have mass
            p_indices = np.arange(L*L)
            q_indices = np.arange(L*L)
            p_coords = np.array(np.unravel_index(p_indices, (L, L))).T
            q_coords = np.array(np.unravel_index(q_indices, (L, L))).T
        
        # Compute cost matrix (distance between all pairs of points)
        # Use POT's built-in distance computation
        if metric == 'euclidean':
            # Compute pairwise Euclidean distances
            M = ot.dist(p_coords, q_coords, metric='euclidean')
        elif metric == 'manhattan':
            M = ot.dist(p_coords, q_coords, metric='cityblock')
        else:
            # Use scipy for other metrics
            from scipy.spatial.distance import cdist
            M = cdist(p_coords, q_coords, metric=metric)
        
        # Solve the optimal transport problem
        # This computes the Wasserstein-1 distance (EMD)
        # method='emd' gives exact solution using network simplex
        
        # For 1D or small problems, use EMD
        if len(p_flat) * len(q_flat) <= 10000:  # Reasonable size for exact EMD
            
            #print(f"{np.sum(p_flat)} --- {np.sum(q_flat)}")
            #if np.sum(q_flat) < 0.01:
                #print(q_flat)
            transport_plan = ot.emd(p_flat, q_flat, M)
        else:
            # For larger problems, use Sinkhorn (regularized OT) for speed
            # Note: Sinkhorn gives an approximation with regularization
            reg = 0.1  # Regularization parameter (adjust as needed)
            transport_plan = ot.sinkhorn(p_flat, q_flat, M, reg)
        
        # Wasserstein distance is the trace of (transport_plan * M)
        distance = np.sum(transport_plan * M)
        
        if return_plan:
            # Reconstruct full transport plan on the original LxL grid
            full_plan = np.zeros((L*L, L*L))
            for i, pi in enumerate(p_indices):
                for j, qj in enumerate(q_indices):
                    full_plan[pi, qj] = transport_plan[i, j]
            return distance, full_plan.reshape(L*L, L*L)
        
        return distance

    L = int(math.sqrt(original_graph_size))
    weights = []
    if important_nodes == []:
        weights = np.ones((L, L))
    else:
        weights = np.zeros(original_graph_size)
        for node in important_nodes:
            weights[node] = 1
        weights = list(weights)
    
    spearman_correlation = scipy.stats.spearmanr(sorted_prop_time_active_o, prop_mod_sorted)
    
    matrix_o = np.reshape(proportions_of_time_active_o, [L, L])
    matrix_m = np.reshape(proportions_of_time_active_m, [L, L])
    #print(matrix_m)
    weight_matrix = np.reshape(weights, [L, L])
    #print(matrix_o)

    wasserstein_dist = 0#wasserstein_2d_grid_pot(matrix_o, matrix_m, metric='manhattan')#scipy.stats.wasserstein_distance(sorted_prop_time_active_o, prop_mod_sorted)
    #print("calculating 2d wasserstein distance")
    jensenshannon_dist = 0
    if np.inf not in matrix_m and np.nan not in matrix_m:
        #print(matrix_m)
        #print(np.sum(matrix_m*weight_matrix))
        #jensenshannon_dist = wasserstein_2d_grid_pot(matrix_o*weight_matrix, matrix_m*weight_matrix, metric='manhattan') #wasserstein_1_distance_2d_grid(matrix_o, matrix_m) #scipy.stats.wasserstein_distance_nd(matrix_o, matrix_m, u_weights=weight_matrix, v_weights=weight_matrix) #weighted_kl_divergence_symmetric(sorted_prop_time_active_o, prop_mod_sorted, weights=weights) #scipy.special.kl_div(sorted_prop_time_active_o, prop_mod_sorted)
        jensenshannon_dist = scipy.spatial.distance.jensenshannon(proportions_of_time_active_o, proportions_of_time_active_m)
    #print("done with it")
    linr = scipy.stats.linregress(sorted_prop_time_active_o, prop_mod_sorted)
    if verbose:
        print(f"{linr.slope}, {linr.intercept}, {linr.pvalue}, {linr.stderr}, {linr.intercept_stderr}")
    #print(prop_orig[0:25])
    #print(a[0:25])
    #x = np.random.rand(len(prop_mod))
    #y = np.random.rand(len(prop_mod))
    if show_fig:
        plt.figure()
        plt.scatter(sorted_prop_time_active_o, prop_mod_sorted)
        x=np.linspace(0,1)#max(max(sorted_prop_time_active_o), max(prop_mod_sorted)))
        plt.suptitle(title)
        
        plt.plot(x, linr.intercept + linr.slope*x, 'r', label='fitted line')
    if save_fig:
        plt.savefig(f"{title}.png", dpi=600)
    if verbose:
        print(f"linear regression R^2: {linr.rvalue**2}")
        print(f"absolute node-by-node activity difference, averaged: {average_activity_comparison}")
        print(f"spearman rho: {spearman_correlation.statistic}")
        #print(f"spearman p-value: {spearman_correlation.pvalue}")
        print(f"2D wasserstein of important sites: {jensenshannon_dist}")
        print(f"2D wasserstein distance: {wasserstein_dist}")
    
    if wass_out and not jensen_out:
        return spearman_correlation.pvalue, linr.rvalue**2, spearman_correlation.statistic, wasserstein_dist
    elif wass_out:
        return spearman_correlation.pvalue, linr.rvalue**2, spearman_correlation.statistic, wasserstein_dist, jensenshannon_dist
    
    return spearman_correlation.pvalue, linr.rvalue**2, spearman_correlation.statistic



def density_compare_memsafe(orig_data, mod_data):
    #get prop_time_active_orig
    #data_arr = [orig_data, mod_data]
    #prop_times_active = np.empty(2)
            
    sum_activity_original = 0
    sum_inactivity_original = 0
    for site in orig_data:

        sum_activity_original += orig_data[site][0]
        sum_inactivity_original += orig_data[site][1]
    
            
    sum_activity_mod = 0
    sum_inactivity_mod = 0

    for site in mod_data:
        
        sum_activity_mod += mod_data[site][0]
        sum_inactivity_mod += mod_data[site][1]
    
    orig_density = sum_activity_original/(sum_activity_original+sum_inactivity_original)
    mod_density = sum_activity_mod/(sum_activity_mod+sum_inactivity_mod)

    return orig_density, mod_density

def get_state_density(G):
    is_active = G.getNodeAttribute("active", int)
    return sum([is_active[u] for u in G.iterNodes()])

def quasistationary_compare0():
    G = generate_square_lattice(128, 128)
    
    G_sdrg = copy_graph(G)
    G_smds = copy_graph(G)
    G_effr = copy_graph(G)

    
    #G_smds = SMDS_sparsifier(G_smds, 0)
    #num_edges = G_smds.numberOfEdges()
    #G_sdrg = sdrg_sparsify_n_edges(G_sdrg, num_edges)
    #G_effr = effective_resistance_sampling_sparsification(G_effr, num_edges)
    
    
    reset_to_distributed_infection(G, 1)
    #reset_to_distributed_infection(G_sdrg, 1)
    #reset_to_distributed_infection(G_smds, 1)
    #reset_to_distributed_infection(G_effr, 1)
    
    print("starting")
    data_dense0 = fast_sparsified_asymptotic_quasistationary_activity_probability(G, t_relax = 5000, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[128, 128], title="original graph")
    print("done getting dense")
    with open("dense2_0.pkl", "wb") as file:
        pickle.dump(data_dense0, file)
    print("saved dense2_0")
    
    reset_to_distributed_infection(G, 1)
    data_dense1 = fast_sparsified_asymptotic_quasistationary_activity_probability(G, t_relax = 5000, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[128, 128], title="original graph")
    print("done getting dense")
    with open("dense2_1.pkl", "wb") as file:
        pickle.dump(data_dense1, file)
    print("saved dense2_1")
    """
    data_sdrg = fast_sparsified_asymptotic_quasistationary_activity_probability(G_sdrg, t_relax = 5000, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[128, 128], title = "sdrg sparsified")
    with open("sdrg0.pkl", "wb") as file:
        pickle.dump(data_sdrg, file)

    data_smds = fast_sparsified_asymptotic_quasistationary_activity_probability(G_smds, t_relax = 5000, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[128, 128], title = "smds sparsified")
    with open("smds0.pkl", "wb") as file:
        pickle.dump(data_smds, file)
    
    data_effr = fast_sparsified_asymptotic_quasistationary_activity_probability(G_effr, t_relax = 5000, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[128, 128], title = "effR sparsified")
    with open("effr0.pkl", "wb") as file:
        pickle.dump(data_effr, file)
    """
    print(" ---------------------- data obtained ---------------------- ")
    compare_data(data_dense0, data_dense1, G.numberOfNodes(), 5000)
    """
    print("sdrg compare")
    compare_data(data_dense, data_sdrg, G.numberOfNodes(), 5000)
    print("smds compare")
    compare_data(data_dense, data_smds, G.numberOfNodes(), 5000)
    print("effr compare")
    compare_data(data_dense, data_effr, G.numberOfNodes(), 5000)
    """


def quasistationary_compare1():
    G = generate_square_lattice(32, 1)
    #G = generate_chain(32)
    #G_sdrg = copy_graph(G)
    #G_smds = copy_graph(G)
    #G_effr = copy_graph(G)

    
    #G_smds = SMDS_sparsifier(G_smds, 0)
    #num_edges = G_smds.numberOfEdges()
    #G_sdrg = sdrg_sparsify_n_edges(G_sdrg, num_edges)
    #G_effr = effective_resistance_sampling_sparsification(G_effr, num_edges)
    
    
    reset_to_distributed_infection(G, 1)
    #reset_to_distributed_infection(G_sdrg, 1)
    #reset_to_distributed_infection(G_smds, 1)
    #reset_to_distributed_infection(G_effr, 1)
    t_relax_set = 2000000
    print("starting")
    data_dense0 = fast_sparsified_asymptotic_quasistationary_activity_probability(G, t_relax = t_relax_set, G_structure="chain", original_graph_size=G.numberOfNodes(), dimensions=[32, 32], title="original graph")
    print("done getting dense")
    with open("dense_qc1run4_0.pkl", "wb") as file:
        pickle.dump(data_dense0, file)
    print("saved dense2_0")
    
    
    
    reset_to_distributed_infection(G, 1)
    
    data_dense1 = fast_sparsified_asymptotic_quasistationary_activity_probability(G, t_relax = t_relax_set, G_structure="chain", original_graph_size=G.numberOfNodes(), dimensions=[32, 32], title="original graph")
    print("done getting dense")
    with open("dense_qc1run4_1.pkl", "wb") as file:
        pickle.dump(data_dense1, file)
    print("saved dense2_1")
    
    """
    data_sdrg = fast_sparsified_asymptotic_quasistationary_activity_probability(G_sdrg, t_relax = 5000, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[128, 128], title = "sdrg sparsified")
    with open("sdrg0.pkl", "wb") as file:
        pickle.dump(data_sdrg, file)

    data_smds = fast_sparsified_asymptotic_quasistationary_activity_probability(G_smds, t_relax = 5000, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[128, 128], title = "smds sparsified")
    with open("smds0.pkl", "wb") as file:
        pickle.dump(data_smds, file)
    
    data_effr = fast_sparsified_asymptotic_quasistationary_activity_probability(G_effr, t_relax = 5000, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[128, 128], title = "effR sparsified")
    with open("effr0.pkl", "wb") as file:
        pickle.dump(data_effr, file)
    """
    print(" ---------------------- data obtained ---------------------- ")
    compare_data(data_dense0, data_dense1, G.numberOfNodes(), t_relax_set)
    
    """
    print("sdrg compare")
    compare_data(data_dense, data_sdrg, G.numberOfNodes(), 5000)
    print("smds compare")
    compare_data(data_dense, data_smds, G.numberOfNodes(), 5000)
    print("effr compare")
    compare_data(data_dense, data_effr, G.numberOfNodes(), 5000)
    """


def quasistationary_compare2():
    L = 8
    t_relax_set = 1000000
    
    G = generate_square_lattice(L, L)
    #G = generate_chain(32)
    G_sdrg = copy_graph(G)
    G_smds = copy_graph(G)
    G_effr = copy_graph(G)

    
    G_smds = SMDS_sparsifier(G_smds, 0)
    num_edges = G_smds.numberOfEdges()
    G_sdrg = sdrg_sparsify(G_sdrg)
    
    G_effr = effective_resistance_sampling_sparsification(G_effr, num_edges)
    
    print("EDGE COUNT ------------- ")
    print(f"base: {G.numberOfEdges()}")
    print(f"sdrg: {G_sdrg.numberOfEdges()}")
    print(f"smds: {G_smds.numberOfEdges()}")
    print(f"effr: {G_effr.numberOfEdges()}")
    
    reset_to_distributed_infection(G, 1)
    reset_to_distributed_infection(G_sdrg, 1)
    reset_to_distributed_infection(G_smds, 1)
    reset_to_distributed_infection(G_effr, 1)
    
    print("starting ----------------")
    data_dense0 = fast_sparsified_asymptotic_quasistationary_activity_probability(G, t_relax = t_relax_set, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L, L], title="original graph 0")
    print("done getting dense0 data")
    with open("dense_qc2run0_0.pkl", "wb") as file:
        pickle.dump(data_dense0, file)
    print("saved dense0")
    
    reset_to_distributed_infection(G, 1)
    
    data_dense1 = fast_sparsified_asymptotic_quasistationary_activity_probability(G, t_relax = t_relax_set, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L, L], title="original graph 1")
    print("done getting dense1 data")
    with open("dense_qc2run0_1.pkl", "wb") as file:
        pickle.dump(data_dense1, file)
    print("saved dense1")
    
    
    data_sdrg = fast_sparsified_asymptotic_quasistationary_activity_probability(G_sdrg, t_relax = t_relax_set, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L, L], title = "sdrg sparsified")
    with open("sdrg_qc2_run0.pkl", "wb") as file:
        pickle.dump(data_sdrg, file)

    data_smds = fast_sparsified_asymptotic_quasistationary_activity_probability(G_smds, t_relax = t_relax_set, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L, L], title = "smds sparsified")
    with open("smds_qc2_run0.pkl", "wb") as file:
        pickle.dump(data_smds, file)
    
    data_effr = fast_sparsified_asymptotic_quasistationary_activity_probability(G_effr, t_relax = t_relax_set, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L, L], title = "effR sparsified")
    with open("effr_qc2_run0.pkl", "wb") as file:
        pickle.dump(data_effr, file)
    
    print(" ---------------------- data obtained ---------------------- ")
    print(" -------------- check quasistationary time -------------- ")
    compare_data(data_dense0, data_dense1, G.numberOfNodes(), t_relax_set)

    
    print(" ----------------------     sdrg compare")
    compare_data(data_dense0, data_sdrg, G.numberOfNodes(), t_relax_set)
    print(" ----------------------     smds compare")
    compare_data(data_dense0, data_smds, G.numberOfNodes(), t_relax_set)
    print(" ----------------------     effr compare")
    compare_data(data_dense0, data_effr, G.numberOfNodes(), t_relax_set)
    


def quasistationary_compare3():
    L = 32
    t_relax_set = 2500000
    
    G = generate_square_lattice(L, L)
    #G = generate_chain(32)
    G_sdrg = copy_graph(G)
    G_smds = copy_graph(G)
    G_effr = copy_graph(G)

    
    G_smds = SMDS_sparsifier(G_smds, 0)
    num_edges = G_smds.numberOfEdges()
    G_sdrg = sdrg_sparsify(G_sdrg)
    
    G_effr = effective_resistance_sampling_sparsification(G_effr, num_edges)
    
    print("EDGE COUNT ------------- ")
    print(f"base: {G.numberOfEdges()}")
    print(f"sdrg: {G_sdrg.numberOfEdges()}")
    print(f"smds: {G_smds.numberOfEdges()}")
    print(f"effr: {G_effr.numberOfEdges()}")
    
    reset_to_distributed_infection(G, 1)
    reset_to_distributed_infection(G_sdrg, 1)
    reset_to_distributed_infection(G_smds, 1)
    reset_to_distributed_infection(G_effr, 1)
    
    print("starting ----------------")
    data_dense0, data_dense1 = fast_quasistationary_activity_probability_checkpoints(G, t_relax = t_relax_set, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L, L], title="original graph 0")
    print("done getting dense0 & 1 data")
    with open("dense_qc3run0_0.pkl", "wb") as file:
        pickle.dump(data_dense0, file)
    print("saved dense0")
    
    
    with open("dense_qc3run0_1.pkl", "wb") as file:
        pickle.dump(data_dense1, file)
    print("saved dense1")
    
    
    data_sdrg = fast_sparsified_asymptotic_quasistationary_activity_probability(G_sdrg, t_relax = t_relax_set, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L, L], title = "sdrg sparsified")
    with open("sdrg_qc2_run0.pkl", "wb") as file:
        pickle.dump(data_sdrg, file)

    data_smds = fast_sparsified_asymptotic_quasistationary_activity_probability(G_smds, t_relax = t_relax_set, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L, L], title = "smds sparsified")
    with open("smds_qc2_run0.pkl", "wb") as file:
        pickle.dump(data_smds, file)
    
    data_effr = fast_sparsified_asymptotic_quasistationary_activity_probability(G_effr, t_relax = t_relax_set, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L, L], title = "effR sparsified")
    with open("effr_qc2_run0.pkl", "wb") as file:
        pickle.dump(data_effr, file)
    
    print(" ---------------------- data obtained ---------------------- ")
    print(" -------------- check quasistationary time -------------- ")
    compare_data(data_dense0, data_dense1, G.numberOfNodes(), t_relax_set)

    
    print(" ----------------------     sdrg compare")
    compare_data(data_dense0, data_sdrg, G.numberOfNodes(), t_relax_set)
    print(" ----------------------     smds compare")
    compare_data(data_dense0, data_smds, G.numberOfNodes(), t_relax_set)
    print(" ----------------------     effr compare")
    compare_data(data_dense0, data_effr, G.numberOfNodes(), t_relax_set)
    

def critical_quasistationary_run(G, original_graph_size, L, filename=""):
    #first we make G critical
    mu_scale = SDRG_crit_point_estimation(G, 30)
    
    G = scale_mu(G, mu_scale)
    print("scaled to critical point")
    #reset_to_distributed_infection(G, 1)
    
    #sparsified_DCP_fast_memsafe(G, t_max=t, original_graph_size=original_graph_size)

    data = fast_dcp_until_quasistationary_memsafe(G,init_time=2, G_structure="lattice", original_graph_size=original_graph_size, title=f"L = {L}, d=2", dimensions=[L,L], spearman_thresh = 0.98, viz=False)
    
    #data = fast_sparsified_asymptotic_quasistationary_activity_probability(G, t_relax = t, G_structure="lattice", original_graph_size=original_graph_size, dimensions=[L, L], title = title)
    with open(f"{filename}.pkl", "wb") as file:
        pickle.dump(data, file)
        
    #vis_lat_advanced_memsafe(data, original_graph_size, t, "lattice", [L, L], title)
    

def gen_plots():
    with open("dense_qc2run0_0.pkl", "rb") as file:
        loaded_data = pickle.load(file)
    with open("dense_qc2run0_1.pkl", "rb") as file:
        loaded_compare = pickle.load(file)

    compare_data(loaded_data, loaded_compare, 8*8, 1000000)



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
    smds_ATES = []
    effR_ATES = []
    
    G_sdrg = copy_graph(G)
    G_smds = copy_graph(G)
    G_effr = copy_graph(G)
    
    G_sdrg = sdrg_sparsify(G_sdrg)
    G_smds = SMDS_sparsifier(G_smds, 0)
    
    
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
        
    sparsification_proportions = np.linspace(0,1,k)#sparsification_range[0], sparsification_range[1], k)
    
    #print(initial_num_components*sparsification_proportions)
    #print((initial_num_components*sparsification_proportions)-4096)
    
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


""" ------------------------------------ UTILITIES ------------------------------------ """
def run_viz_step(G, t):
    pass

#def num_active_in_component(G, node_id):

def open_data(filename):
    with open(f"{filename}.pkl", "rb") as file:
        loaded_data = pickle.load(file)
        return loaded_data

def save_data(data, filename):
    with open(f"{filename}.pkl", "wb") as file:
        pickle.dump(data, file)

def collate_fast_dcp_data(data0, data1, t):
    #TODO This method will slowly "lose" time because the DCP sims always run a tiny bit longer than their t_max. This shouldn't be a big issue but maybe something to fix. 
    d_out = data0
    for site in range(len(d_out)):
        #print(site)
        d_out[site].append(t)
        for transition in data1[site][1:]:
            d_out[site].append(transition + t)
    return d_out
   

    
def collate_fast_dcp_memsafe_data(data0, data1):
    #TODO This method will slowly "lose" time because the DCP sims always run a tiny bit longer than their t_max. This shouldn't be a big issue but maybe something to fix. 
    d_out = data0
    for site in range(len(d_out)):
        #print(site)
        d_out[site][0] += data1[site][0]
        d_out[site][1] += data1[site][1]
        d_out[site][2] = d_out[site][0] + d_out[site][1] # this value shouldn't matter as it has no impact on the next simulation
        d_out[site][3] = False #we indicate a final transition from active -> inactive @ t_max
    return d_out
   
        

def get_max_omega(G):
    mu = G.getNodeAttribute("mu", float)
    omega = 0
    for u in G.iterNodes():
        mu_val = mu[u]
        if mu_val > omega:
            omega = mu_val
    for u,v in G.iterEdges():
        w = G.weight(u,v)
        if w > omega:
            omega = w
    return omega

def get_list_of_edge_components(G, eid):
    #t0 = time.time_ns()
    
    #edge components stores exclusively tuples of edges which comprise the edge with id = eid
    edge_components = G.getEdgeAttribute("e_comp", str)
    l = edge_components[eid]
    expr = list(set(eval("[" + l.replace("_", ",") + "]")))
    #out = []
    #print(f"gloec --------: {(time.time_ns()-t0)/1000000000}")
    #t0 = time.time_ns()
    #for edge in l:
    #    tup = eval(edge)
    #    out.append(tup)
    #print(f"gloec --------: {(time.time_ns()-t0)/1000000000}")
    #t0 = time.time_ns()
    return expr

def add_e_comp(G):
    comp = G.attachEdgeAttribute("e_comp", str)
    
    for edge in G.iterEdges():
        comp[edge] = f"{edge}"

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

def number_connected_components(G):
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    print(f"{len(cc.getComponentSizes())} components: {cc.getComponentSizes()}")

def complexity_score(G):
    score = 0
    for u in G.iterNodes():
        score += G.degree(u)
    return score


def fully_text_graph(G, filename):
    n_nodes = G.numberOfNodes()
    
    is_active = G.getNodeAttribute("active", int)
    healing_factor = G.getNodeAttribute("mu", float)
    components = G.getNodeAttribute("components", str)
    edge_components = G.getEdgeAttribute("e_comp", str)
    mu_components = G.getNodeAttribute("mu_comp", str)
    lambda_components = G.getEdgeAttribute("lambda_comp", str)
    
    lines_to_write = [f"G = nk.graph.Graph(n={G.numberOfNodes()}, weighted=True, edgesIndexed=True)", 
                      "is_active = G.attachNodeAttribute('active', int)",
                      "healing_factor = G.attachNodeAttribute('mu', float)",
                      "components = G.attachNodeAttribute('components', str)",
                      "edge_components = G.attachEdgeAttribute('e_comp', str)",
                      "mu_components = G.attachNodeAttribute('mu_comp', str)",
                      "lambda_components = G.attachEdgeAttribute('lambda_comp', str)",
                      "print('doing something')"
                      ]
    
    for u in G.iterNodes():
        lines_to_write.append(f"healing_factor[{u}] = {healing_factor[u]}")
        lines_to_write.append(f"is_active[{u}] = {is_active[u]}")
        lines_to_write.append(f"components[{u}] = '{components[u]}'")
        lines_to_write.append(f"mu_components[{u}] = '{mu_components[u]}'")
    
    for edge in G.iterEdges():
        lines_to_write.append(f"G.addEdge({edge[0]}, {edge[1]}, w={G.weight(edge[0], edge[1])})")
        lines_to_write.append(f"eid = {G.edgeId(edge[0], edge[1])}") # eid = 4
        eid = G.edgeId(edge[0], edge[1])
        comps = edge_components[eid]
        lines_to_write.append(f"edge_components[eid] = '{edge_components[eid]}'") #edge_components[4] = 
        lines_to_write.append(f"lambda_components[eid] = '{lambda_components[eid]}'")
        
    
    with open(f"{filename}.txt", "w") as file:
        
        for line in lines_to_write:
            file.write(line + "\n")  # Adds a newline character after each string
    
def read_fully_text_graph(filename):
    with open(f"{filename}.txt", "r") as file:
        for line in file:
            exec(line)
            
def copy_graph(G_in):
    G = nk.graph.Graph(n=G_in.numberOfNodes(), weighted=True, edgesIndexed=True)
    
    in_activity = G_in.getNodeAttribute("active", int)
    in_mu = G_in.getNodeAttribute("mu", float)
    in_comp = G_in.getNodeAttribute("components", str)
    in_edge_comps = G_in.getEdgeAttribute("e_comp", str)
    in_mu_comps = G_in.getNodeAttribute("mu_comp", str)
    in_lam_comps = G_in.getEdgeAttribute("lambda_comp", str)
    in_infection_progenitor = G_in.getNodeAttribute("inf_prog", int)
    
    is_active = G.attachNodeAttribute("active", int)
    healing_factor = G.attachNodeAttribute("mu", float)
    components = G.attachNodeAttribute("components", str)
    edge_components = G.attachEdgeAttribute("e_comp", str)
    mu_components = G.attachNodeAttribute("mu_comp", str)
    lambda_components = G.attachEdgeAttribute("lambda_comp", str)
    infection_progenitor = G.attachNodeAttribute("inf_prog", int)
    #for site in G.iterNodes():
    #    infection_progenitor[site] = site
            
    #add disorder to healing values too if desired
    for edge in G_in.iterEdges():
        G.addEdge(edge[0], edge[1], w=G_in.weight(edge[0], edge[1]))
        old_eid = G_in.edgeId(edge[0], edge[1])
        eid = G.edgeId(edge[0], edge[1])
        edge_components[eid] = in_edge_comps[old_eid]
        lambda_components[eid] = in_lam_comps[old_eid]

    for u in G.iterNodes():
        healing_factor[u] = in_mu[u]
        is_active[u] = in_activity[u]
        components[u] = in_comp[u]
        mu_components[u] = in_mu_comps[u]
        infection_progenitor[u] = in_infection_progenitor[u]
    return G

def append_graphs(G0, G1):
    #G0 = copy_graph(G0_in)
    #G1 = copy_graph(G1_in)
    
    G = nk.graph.Graph(n=G0.numberOfNodes()*2, weighted=True, edgesIndexed=True)
    L_sq = int(G0.numberOfNodes())
    
    in_activity0 = G0.getNodeAttribute("active", int)
    in_mu0 = G0.getNodeAttribute("mu", float)
    in_comp0 = G0.getNodeAttribute("components", str)
    in_edge_comps0 = G0.getEdgeAttribute("e_comp", str)
    in_mu_comps0 = G0.getNodeAttribute("mu_comp", str)
    in_lam_comps0 = G0.getEdgeAttribute("lambda_comp", str)
    in_infection_progenitor0 = G0.getNodeAttribute("inf_prog", int)
    #visualize(G0)
    #print(list(in_edge_comps0))
    #print(f"------ len = {len(list(in_edge_comps0))}")
    in_activity1 = G1.getNodeAttribute("active", int)
    in_mu1 = G1.getNodeAttribute("mu", float)
    in_comp1 = G1.getNodeAttribute("components", str)
    in_edge_comps1 = G1.getEdgeAttribute("e_comp", str)
    in_mu_comps1 = G1.getNodeAttribute("mu_comp", str)
    in_lam_comps1 = G1.getEdgeAttribute("lambda_comp", str)
    in_infection_progenitor1 = G1.getNodeAttribute("inf_prog", int)
    
    is_active = G.attachNodeAttribute("active", int)
    healing_factor = G.attachNodeAttribute("mu", float)
    components = G.attachNodeAttribute("components", str)
    edge_components = G.attachEdgeAttribute("e_comp", str)
    mu_components = G.attachNodeAttribute("mu_comp", str)
    lambda_components = G.attachEdgeAttribute("lambda_comp", str)
    infection_progenitor = G.attachNodeAttribute("inf_prog", int)
    
    for edge in G0.iterEdges():
        #print(edge)
        G.addEdge(edge[0], edge[1], w=G0.weight(edge[0], edge[1]))
        #print(G0.weight(edge[0], edge[1]))
        eid_old = G0.edgeId(edge[0], edge[1])
        eid_new = G.edgeId(edge[0], edge[1])
        #print(f"{eid_old} -> {eid_new}")
        #print(in_edge_comps0[eid_old])
        edge_components[eid_new] = in_edge_comps0[eid_old]
        lambda_components[eid_new] = in_lam_comps0[eid_old]
    
    #print("-------- G1 ------------")
    
    for edge in G1.iterEdges():
        G.addEdge(edge[0]+L_sq, edge[1]+L_sq, w=G1.weight(edge[0], edge[1]))
        eid_old = G1.edgeId(edge[0], edge[1])
        eid_new = G.edgeId(edge[0]+L_sq, edge[1]+L_sq)
        edge_components[eid_new] = in_edge_comps1[eid_old]
        lambda_components[eid_new] = in_lam_comps1[eid_old]

    #print(list(edge_components))
    #print(len(list(edge_components)))
    for u in G0.iterNodes():
        healing_factor[u] = in_mu0[u]
        is_active[u] = in_activity0[u]
        components[u] = in_comp0[u]
        mu_components[u] = in_mu_comps0[u]
        infection_progenitor[u] = in_infection_progenitor0[u]
    
    for u in G1.iterNodes():
        healing_factor[u + L_sq] = in_mu1[u]
        is_active[u + L_sq] = in_activity1[u]
        #print('n' + str(int(in_mu_comps1[u][1:])+L_sq))
        components[u + L_sq] = str(int(in_comp1[u])+L_sq)
        mu_components[u + L_sq] = 'n' + str(int(in_mu_comps1[u][1:])+L_sq)
        infection_progenitor[u + L_sq] = in_infection_progenitor1[u]
    
    return G

def get_sublist_with_value(val, superlist):
    for sublist in superlist:
        if val in sublist:
            return sublist
    raise ValueError(f"{val} is not in any of the sublists either!")

def get_index_of_sublist_with_value(val, superlist):
    for i in range(len(superlist)):
        sublist = superlist[i]
        if val in sublist:
            return i
    raise ValueError(f"{val} is not in any of the sublists either!")
    
def print_graph_values(G, just_first_few = True):
    healing_factor = G.getNodeAttribute("mu", float)
    num_edges = G.numberOfEdges()
    num_nodes = G.numberOfNodes()
    print(f"G has {num_edges} edges and {num_nodes} nodes \n------------------")
    
    if just_first_few:
        prints = 0
        for u in G.iterNodes():
            if prints < 10:
                print(f"node {u} has mu = {healing_factor[u]}")
                prints += 1
        
        print(f"... {num_nodes - 10} more unprinted node values")
        
        for u,v in G.iterEdges():
            if prints < 20:
                print(f"edge from {u} to {v} has lambda = {G.weight(u,v)}")
                prints += 1
        print(f"... {num_edges - 10} more unprinted edge vals")
    else:
        for u in G.iterNodes():
            print(f"node {u} has mu = {healing_factor[u]}")
        for u,v in G.iterEdges():
            print(f"edge from {u} to {v} has lambda = {G.weight(u,v)}")
    
    if num_edges > 0:
        lambda_avg = (G.totalEdgeWeight()/num_edges)
        print(f"The average lambda = {lambda_avg}")
        
def remove_edge_safe(G, u, v):
    if G.hasEdge(u,v):
        G.removeEdge(u, v)    
                                                                                                                                                                           
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
    
def visualize_lattice(G, L):
    
    #Draw a Networkit graph with nodes arranged in a square lattice.
    
    # Assign each node a lattice position.
    positions = {}

    for i, node in enumerate(G.iterNodes()):
        row = i // L
        col = i % L

        # Center the lattice around (0, 0)
        x = col - (L - 1) / 2
        y = -(row - (L - 1) / 2)

        positions[node] = (x, y)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw all edges
    for u, v in G.iterEdges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="gray",
            linewidth=2.5,
            zorder=1
        )

    # Draw all nodes
    xs = [positions[node][0] for node in positions]
    ys = [positions[node][1] for node in positions]

    ax.scatter(
        xs,
        ys,
        s=120,
        color="steelblue",
        edgecolors="black",
        zorder=2
    )

    # Label nodes with their Networkit IDs
    for node, (x, y) in positions.items():
        ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            zorder=3
        )

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
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
    
def vis_lat_advanced(data, original_graph_size, relaxation_time, G_structure, dimensions, title, highlight_nodes=[], save_fig = False):
    #matrix = np.matrix(data.to_numpy())
    
    test_trans = data[0]
    test_trans.append(relaxation_time)
    total_time_active_test = 0
    for i in range(1, len(test_trans), 2):
        total_time_active_test += test_trans[i] - test_trans[i-1]
    #total_time_active_list[site] = total_time_active
    print(total_time_active_test)
    #print(matrix.shape)
    total_time_active_list = np.empty(original_graph_size)
    for site in data:
        trans_times = data[site]
        trans_times.append(relaxation_time)
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
        plt.figure()
        fig, ax = plt.subplots()
        plt.bar(range(original_graph_size), proportions_of_time_active, width=1, linewidth=0)
        fig.suptitle(title, fontsize=12)
        
        if save_fig:
            plt.savefig(f"{title}.png", dpi=600)
        plt.show()
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
        
        plt.figure()
        
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, Z, cmap='viridis', vmin=0, vmax=1)

        #fig.tight_layout()

        #fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
        
        # plot just the positive data and save the
        # color "mappable" object returned by ax1.imshow
        pos = ax.imshow(Z, cmap='viridis', interpolation='none')
        
        # add the colorbar using the figure's method,
        # telling which mappable we're talking about and
        # which Axes object it should be near
        fig.colorbar(pos, ax=ax, spacing='uniform')
        fig.suptitle(title, fontsize=12)
        if save_fig:
            plt.savefig(f"{title}.png", dpi=600)
        plt.show()
   

def vis_lat_advanced_memsafe(data, original_graph_size, relaxation_time, G_structure, dimensions, title, nodes_to_highlight=[], save_fig=False):
    #matrix = np.matrix(data.to_numpy())
    
    #print(matrix.shape)
    total_time_active_list = np.empty(original_graph_size*2)
    for site in data:
        
        if site == 0:
            #print(data[site][0])
            pass
        
        total_time_active_list[site] = data[site][0]
    
    proportions_of_time_active = total_time_active_list/relaxation_time
    #print(proportions_of_time_active_listform)
    #proportions_of_time_active = [u[0,0] for u in proportions_of_time_active_listform]
    #print(proportions_of_time_active)
    #print(proportions_of_time_active)
    if G_structure == "chain":
        plt.figure()
        fig, ax = plt.subplots()
        plt.bar(range(original_graph_size), proportions_of_time_active, width=1, linewidth=0)
        fig.suptitle(title, fontsize=12)
        if save_fig:
            plt.savefig(f"{title}.png", dpi=600)
        else:
            plt.show()
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
        
        plt.figure()
        
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, Z, cmap='viridis', vmin=0, vmax=1)
        L = math.sqrt(original_graph_size)
        for imp_node in nodes_to_highlight:
            y_coord = imp_node % w
            x_coord = imp_node // h
        
            ax.scatter([x_coord], [y_coord], color='red', edgecolor='red', s=200/L, zorder=5, label='Point Overlay')

        #fig.tight_layout()

        #fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
        
        # plot just the positive data and save the
        # color "mappable" object returned by ax1.imshow
        pos = ax.imshow(Z, cmap='viridis', interpolation='none')
        
        # add the colorbar using the figure's method,
        # telling which mappable we're talking about and
        # which Axes object it should be near
        fig.colorbar(pos, ax=ax, spacing='uniform')
        fig.suptitle(title, fontsize=12)
        if save_fig:
            plt.savefig(f"{title}.png", dpi=600)
        else:
            plt.show()
    if G_structure == "network":
       
        h = math.ceil(math.sqrt(dimensions[0]))
        w = h

        Z = np.zeros((h,w))

        #first build a bunch of lines/rows
        for y in range(h):
            for x in range(w):  
                Z[x,y] = proportions_of_time_active[x+(y*w)]
        
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        plt.figure()
        
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, Z, cmap='viridis', vmin=0, vmax=1)
        L = math.sqrt(original_graph_size)
        for imp_node in nodes_to_highlight:
            y_coord = imp_node % w
            x_coord = imp_node // h
        
            ax.scatter([x_coord], [y_coord], color='red', edgecolor='red', s=200/L, zorder=5, label='Point Overlay')
        
        #fig.tight_layout()
        
        #fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
        
        # plot just the positive data and save the
        # color "mappable" object returned by ax1.imshow
        pos = ax.imshow(Z, cmap='viridis', interpolation='none')
        
        # add the colorbar using the figure's method,
        # telling which mappable we're talking about and
        # which Axes object it should be near
        fig.colorbar(pos, ax=ax, spacing='uniform')
        fig.suptitle(title, fontsize=12)
        if save_fig:
            plt.savefig(f"{title}.png", dpi=600)
        else:
            plt.show()
 
def vis_clusters(G_in, L, G_structure, title):
    G = copy_graph(G_in)
    clusters = get_neil_output(G, verbose=False)
    #matrix = np.matrix(data.to_numpy())
    
    #print(matrix.shape)
    
    num_colors = 5
    
    #print(proportions_of_time_active_listform)
    #proportions_of_time_active = [u[0,0] for u in proportions_of_time_active_listform]
    #print(proportions_of_time_active)
    #print(proportions_of_time_active)
    if G_structure == "chain":
        plt.figure()
        fig, ax = plt.subplots()
        #plt.bar(range(original_graph_size), proportions_of_time_active, width=1, linewidth=0)
        fig.suptitle(title, fontsize=12)
        plt.show()
    if G_structure == "lattice":
        # same code as vis_lat
        
        
        Z = np.zeros((L, L, 3))
        colors = mcolors.CSS4_COLORS
        #print(colors)
        for cluster in clusters:
            r = np.random.randint(0, 147)
            rand_color = colors[list(colors)[r]]
            
            cluster_color = mcolors.to_rgb(rand_color)
            #print(cluster_color)
            for node in cluster:
                y = node % L
                x = node // L
                Z[x,y] = cluster_color
        #first build a bunch of lines/rows
        
        
        x = np.arange(L)
        y = np.arange(L)
        X, Y = np.meshgrid(x, y)
        
        plt.figure()
        
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, Z)
                

        fig.suptitle(title, fontsize=12)
        plt.show()
   
 
def vis_given_clusters(clusters, L, G_structure, title, ignore_singletons = False):
    #G = copy_graph(G_in)
    #clusters = get_neil_output(G, verbose=False)
    #matrix = np.matrix(data.to_numpy())
    
    #print(matrix.shape)
    
    #num_colors = 5
    
    #print(proportions_of_time_active_listform)
    #proportions_of_time_active = [u[0,0] for u in proportions_of_time_active_listform]
    #print(proportions_of_time_active)
    #print(proportions_of_time_active)
    if G_structure == "chain":
        plt.figure()
        fig, ax = plt.subplots()
        #plt.bar(range(original_graph_size), proportions_of_time_active, width=1, linewidth=0)
        fig.suptitle(title, fontsize=12)
        plt.show()
    if G_structure == "lattice":
        # same code as vis_lat
        
        
        Z = np.zeros((L, L, 3))
        colors = mcolors.CSS4_COLORS
        #print(colors)
        count = 0
        for cluster in clusters:
            r = np.random.randint(0, 147)
            if len(cluster) < 3 and ignore_singletons:
                r = 0
            else:
                count+=1
                r = count % 146
                
            rand_color = colors[list(colors)[r]]
            
            
            cluster_color = mcolors.to_rgb(rand_color)
            
            #print(cluster_color)
            for node in cluster:
                y = node // L
                x = node % L
                Z[x,y] = cluster_color
        #first build a bunch of lines/rows
        
        
        x = np.arange(L)
        y = np.arange(L)
        X, Y = np.meshgrid(x, y)
        
        plt.figure()
        
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, Z)
                

        fig.suptitle(title, fontsize=12)
        plt.show()
   
 
def completeness(G):
    if G.numberOfNodes() > 0:
        return G.numberOfEdges()/(G.numberOfNodes()**2)
    else:
        return 1
    
def vis_dcp(active_nodes, dimensions = [1, 1], title = ""):
    #matrix = np.matrix(data.to_numpy())
    
    
    
    max_ind = dimensions[0]*dimensions[1]
    
    active_nodes_list = np.zeros(max_ind)
    
    for node in active_nodes:
        active_nodes_list[node] = 1
    
    # same code as vis_lat
    h = dimensions[0]
    w = dimensions[1]
    
    Z = np.zeros((h,w))
    
    #first build a bunch of lines/rows
    for y in range(h):
        for x in range(w):  
            Z[x,y] = active_nodes_list[x+(y*w)]
    
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    plt.figure()
    
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, Z, cmap='viridis', vmin=0, vmax=1)
            

    #fig.tight_layout()

    #fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
    
    # plot just the positive data and save the
    # color "mappable" object returned by ax1.imshow
    pos = ax.imshow(Z, cmap='viridis', interpolation='none')
    
    # add the colorbar using the figure's method,
    # telling which mappable we're talking about and
    # which Axes object it should be near
    fig.colorbar(pos, ax=ax, spacing='uniform')
    fig.suptitle(title, fontsize=12)
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

#def fast_random_choose_weighted(s, weights, min_val, max_val):
def exp_fit(p1, p2, p3):
    def exponential_model(x, a, b, c):
        return a * np.exp(b * x) + c
    
    x_data = np.array([p1[0], p2[0], p3[0]])
    y_data = np.array([p1[1], p2[1], p3[1]])
    
    initial_guess = [1.0, -0.01, 5.0]
    
    optimized_parameters, covariance = curve_fit(
        exponential_model, 
        x_data, 
        y_data, 
        p0=initial_guess
    )
    
    a_fit, b_fit, c_fit = optimized_parameters
    print(f"Fitted Equation: y = {a_fit:.4f} * e^({b_fit:.4f} * x) + {c_fit:.4f}")
    return a_fit, b_fit, c_fit

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

def test2():
    t0 = time.time_ns()/1000000000
    a = np.random.random()*5
    b = np.random.random()*5
    c = np.random.random()*5
    for i in range(151139951):
        
        
        k = a + b + c
        
    print(time.time_ns()/1000000000 - t0)

def test3(G):
    orig = []
    news = []
    for i in range(5):
        reset_to_distributed_infection(G, 1)
        t0 = time.time_ns()/1000000000
        sparsified_DCP_fast_memsafe(G, t_max = 250, original_graph_size=G.numberOfNodes())
        t1 = time.time_ns()/1000000000
        print(f"new: {t1 - t0}")
        orig.append(t1 - t0)
    for i in range(5):
        reset_to_distributed_infection(G, 1)
        t0 = time.time_ns()/1000000000
        sparsified_DCP_fast_memsafe_s(G, t_max = 250, original_graph_size=G.numberOfNodes())
        t1 = time.time_ns()/1000000000
        print(f"old: {t1 - t0}")
        news.append(t1 - t0)
        
    print("---------------")
    print(f"new: {np.average(orig)}")
    print(f"old: {np.average(news)}")
"""
To keep connected; track the maximum strength connection of each site when it is decimated and add it back to keep things connected
yes we are just on the critical point 


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

#G = generate_square_lattice(64, 64)

def test8(L, seed):
    np.random.seed(seed)#int(time.time_ns()/10000000000000))
    
    G_init = generate_square_lattice(L, L)
    data_orig, t_orig = fast_dcp_until_quasistationary_memsafe(G_init, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} control", return_time=True, spearman_thresh=0.98)
    
    gsmds = semi_metric_backbone(G_init)
    data_smds, t_smds = fast_dcp_until_quasistationary_memsafe(gsmds, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} smds", return_time=True, spearman_thresh=0.98)
    print("============================")
    print("===== Final Data Stuff =====")
    print("============================")
    print(f"smds graph has {gsmds.numberOfEdges()} edges")
    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data_smds, G_init.numberOfNodes(), t_orig, t_smds, title=f"ctrl v.s. smds sparsified network")


def test9():
    pass

def test10(L, seed):
    
    np.random.seed(seed)#int(time.time_ns()/10000000000000))
    
    G_init = generate_square_lattice(L, L)
    increments = np.linspace(0,0.9, 5)
    gsp_list = sdrg_sparsify_partial(G_init, 0, True, local_max=True, return_incremental_sparsifications=True, increments = increments)
    return gsp_list
        
def compare_partial_sparsifications(L, seed = -1, lm = True, add_title_data="partial", version = 1):
    
    #by default, randomize
    if seed == -1:
        seed = int((time.time_ns()/100)%10000000)
        
    np.random.seed(seed)#int(time.time_ns()/10000000000000))
    
    G_init = generate_square_lattice(L, L)
    
    mu_scale = SDRG_crit_point_estimation(G_init, 20)
    
    G_init = scale_mu(G_init, mu_scale)
    #smds = semi_metric_backbone(G_init)
    #print(smds.numberOfEdges())
    increments = np.linspace(0,0.75, 2)
    gsp_list, best_sparsification_percentage = sdrg_sparsify_partial(G_init, 0, True, local_max=lm, return_incremental_sparsifications=True, increments = increments, use_multiplicity=True, get_minimal_sparsification=True)
    
    increments = np.append(increments, best_sparsification_percentage)
    
    i = 0
    spearman_thresh = 0.99
    data_orig, t_orig = fast_dcp_until_quasistationary_memsafe(G_init, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} control {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_orig, f"L{L}x{L}_s{seed}_control_{add_title_data}_mk1")
    #print(increments)
    #print(len(gsp_list))
    #print("===========================")
    times = [t_orig]
    data_list =[data_orig]
    for G in gsp_list:
        
        mu_scale = SDRG_crit_point_estimation(G, 20)
        
        G = scale_mu(G, mu_scale)
        
        data, t = fast_dcp_until_quasistationary_memsafe(G, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} sdrg {increments[i]*100}% {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        times.append(t)
        data_list.append(data)
        save_data(data, f"L{L}x{L}_s{seed}_sdrg_{int(increments[i]*100)}p_{add_title_data}_mk1")
        i += 1
    print("- running smds for comparison -")
    gsmds = SMDS_to_n_edges(G_init, gsp_list[len(gsp_list)-1].numberOfEdges())
    data_smds, t_smds = fast_dcp_until_quasistationary_memsafe(gsmds, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} smds", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)

    
    print("============================")
    print("===== Final Data Stuff =====")
    print("============================")
    
    i = 0
    for G in gsp_list:
        print(f"graph at {increments[i]} has {G.numberOfEdges()} edges")
        
        i+=1
    for data_ind in range(len(data_list)-1):
        print(f"- {increments[data_ind]*100} stats -")
        sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_list[0], data_list[data_ind+1], G.numberOfNodes(), times[0], times[data_ind+1], title=f"ctrl v.s. {increments[data_ind]*100}% sparsified network", save_fig=True)
    
    print("- smds stats -")
    
    print(f"smds graph has {gsmds.numberOfEdges()} edges")
    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data_smds, G_init.numberOfNodes(), t_orig, t_smds, title=f"ctrl v.s. smds sparsified network", save_fig=True)

      
def compare_min_partial_sparsifications(L, seed = -1, lm = True, add_title_data="partial", version = 1):
    
    #by default, randomize
    if seed == -1:
        seed = int((time.time_ns()/100)%10000000)
        
    np.random.seed(seed)#int(time.time_ns()/10000000000000))
    
    G_init = generate_square_lattice(L, L)
    
    mu_scale = SDRG_crit_point_estimation(G_init, 20)
    
    G_init = scale_mu(G_init, mu_scale)
    #smds = semi_metric_backbone(G_init)
    #print(smds.numberOfEdges())
    increments = np.empty(0)
    gsp_list, best_sparsification_percentage = sdrg_sparsify_partial(G_init, 0, True, local_max=lm, return_incremental_sparsifications=True, use_multiplicity=True, get_minimal_sparsification=True)
    
    increments = np.append(increments, best_sparsification_percentage)
    
    i = 0
    spearman_thresh = 0.99
    data_orig, t_orig = fast_dcp_until_quasistationary_memsafe(G_init, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} control {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_orig, f"L{L}x{L}_s{seed}_control_{add_title_data}_mk1")
    #print(increments)
    #print(len(gsp_list))
    #print("===========================")
    times = [t_orig]
    data_list =[data_orig]
    for G in gsp_list:
        
        mu_scale = SDRG_crit_point_estimation(G, 20)
        
        G = scale_mu(G, mu_scale)
        
        data, t = fast_dcp_until_quasistationary_memsafe(G, G_structure="lattice", original_graph_size=G.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} sdrg {increments[i]*100}% {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        times.append(t)
        data_list.append(data)
        save_data(data, f"L{L}x{L}_s{seed}_sdrg_{int(increments[i]*100)}p_{add_title_data}_mk1")
        i += 1
    print("- running smds for comparison -")
    gsmds = SMDS_to_n_edges(G_init, gsp_list[len(gsp_list)-1].numberOfEdges())
    data_smds, t_smds = fast_dcp_until_quasistationary_memsafe(gsmds, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} smds", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_smds, f"L{L}x{L}_s{seed}_smds_{add_title_data}")
    
    print("============================")
    print("===== Final Data Stuff =====")
    print("============================")
    
    i = 0
    for G in gsp_list:
        print(f"graph at {increments[i]} has {G.numberOfEdges()} edges")
        
        i+=1
    for data_ind in range(len(data_list)-1):
        print(f"- {increments[data_ind]*100} stats -")
        sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_list[0], data_list[data_ind+1], G.numberOfNodes(), times[0], times[data_ind+1], title=f"ctrl v.s. {increments[data_ind]*100}% sparsified network", save_fig=True)
    
    print("- smds stats -")
    
    print(f"smds graph has {gsmds.numberOfEdges()} edges")
    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data_smds, G_init.numberOfNodes(), t_orig, t_smds, title=f"ctrl v.s. smds sparsified network", save_fig=True)

    


def test12():
    np.random.seed(25)#int(time.time_ns()/10000000000000))
    
    G_init = generate_square_lattice(16, 16)
    increments = np.linspace(0.1,0.9, 5)
    gsp = sdrg_sparsify_partial(G_init, 0, True, local_max=True, return_incremental_sparsifications=False, increments = increments)
    i = 0
    
    #data_orig = fast_dcp_until_quasistationary_memsafe(G_init, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[8,8], title="8x8 control", spearman_thresh=0.98)
    #save_data(data_orig, "L32x32_control_mk1")
    
    for increment in increments:
        gsp = sdrg_sparsify_partial(G_init, increment, True, local_max=True, return_incremental_sparsifications=False, increments = increments)
        data = fast_dcp_until_quasistationary_memsafe(gsp, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[8,8], title=f"8x8 sdrg {increment*100}%", spearman_thresh=0.98)
        #save_data(data, f"L32x32_sdrg_{increments[i]*100}percent_mk1")
        i += 1

def test_sdrg_backbone(L, lm = True, multiplicity=False):
    t0 = time.time()
    seed = int((time.time_ns()/100)%10000000)
    np.random.seed(seed)
    add_title_data = "st_sdrg"
    G_init = generate_square_lattice(L, L)
    
    mu_scale = SDRG_crit_point_estimation(G_init, 20)
    
    G_init = scale_mu(G_init, mu_scale)
    
    G_sdrg = sdrg_sparsify_partial(G_init, 0, True, local_max=lm, return_incremental_sparsifications=False, use_multiplicity=multiplicity)
    spearman_thresh = 0.98
    data_orig, t_orig = fast_dcp_until_quasistationary_memsafe(G_init, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} control {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_orig, f"L{L}x{L}_s{seed}_control_{add_title_data}")
    
    imp_nodes = []#get_nodes_of_largest_k_clusters(G, 3)
    for site in data_orig:
        #print(ctrl[site])
        if data_orig[site][0] > t_orig * 0.9:
            imp_nodes.append(site)
    
    
    print("- running simulation on sdrg network -")
    mu_scale = SDRG_crit_point_estimation(G_sdrg, 20)
    
    G_sdrg = scale_mu(G_sdrg, mu_scale)
    
    data, t_sdrg = fast_dcp_until_quasistationary_memsafe(G_sdrg, G_structure="lattice", original_graph_size=G_sdrg.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} sdrgFull {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data, f"L{L}x{L}_s{seed}_sdrgFull_{add_title_data}")
    n_samples = G_sdrg.numberOfEdges()
    
    print("- running smds for comparison -")
    gsmds = semi_metric_backbone(G_init) #SMDS_to_n_edges(G_init, n_samples)
    data_smds, t_smds = fast_dcp_until_quasistationary_memsafe(gsmds, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} smds {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_smds, f"L{L}x{L}_s{seed}_smds_{add_title_data}")
    
    
    
    
    """
    print("- running STsdrg for comparison -")
    G_stsdrg = shortest_tree_sdrg(G_init, n_samples)
    
    data_stsdrg, t_stsdrg = fast_dcp_until_quasistationary_memsafe(G_stsdrg, G_structure="lattice", original_graph_size=G_stsdrg.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} ST_SDRG {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_stsdrg, f"L{L}x{L}_s{seed}_STsdrg_{add_title_data}")
    """
    print("- running effR for comparison -")
    geffr = copy_graph(G_init)
    geffr = effective_resistance_sampling_sparsification(geffr, n_samples)
    data_effr, t_effr = fast_dcp_until_quasistationary_memsafe(geffr, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} effr {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_effr, f"L{L}x{L}_s{seed}_effr_{add_title_data}")
    
    print("- running uniform for comparison -")
    guni = copy_graph(G_init)
    guni = uniform_sampling_sparsification(guni, n_samples) 
    data_uni, t_uni = fast_dcp_until_quasistationary_memsafe(geffr, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} uniform {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_uni, f"L{L}x{L}_s{seed}_uniform_{add_title_data}")
    
    print("- running weight sampling for comparison -")
    gweight = copy_graph(G_init)
    gweight = weight_sampling_sparsification(gweight, n_samples) #uniform_sampling_sparsification(guni, n_samples) 
    data_weight, t_weight = fast_dcp_until_quasistationary_memsafe(gweight, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} weighted {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_weight, f"L{L}x{L}_s{seed}_weightBased_{add_title_data}")
    
    print("- running thresholding for comparison -")
    gthresh = copy_graph(G_init)
    gthresh =  threshold_sparsification(gthresh, n_samples) #weight_sampling_sparsification(gweight, n_samples) #uniform_sampling_sparsification(guni, n_samples) 
    data_thresh, t_thresh = fast_dcp_until_quasistationary_memsafe(gthresh, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} thresh {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_thresh, f"L{L}x{L}_s{seed}_thresholding_{add_title_data}")
    
    print("============================")
    print("===== Final Data Stuff =====")
    print("============================")
    
    print("- sdrg stats -")
    print(f"sdrg graph has {G_sdrg.numberOfEdges()} edges")

    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data, G_sdrg.numberOfNodes(), t_orig, t_sdrg, title=f"ctrl v.s. sdrg", important_nodes=imp_nodes, save_fig=True)
    
    
    """
    print("- stsdrg stats -")
    print(f"stsdrg graph has {G_stsdrg.numberOfEdges()} edges")

    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data_stsdrg, G_stsdrg.numberOfNodes(), t_orig, t_stsdrg, title=f"ctrl v.s. STsdrg", important_nodes=imp_nodes, save_fig=True)
    """
    
    
    print("- smds stats -")
    print(f"smds graph has {gsmds.numberOfEdges()} edges")
    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data_smds, gsmds.numberOfNodes(), t_orig, t_smds, title=f"ctrl v.s. smds", important_nodes=imp_nodes, save_fig=True)
    
    
    print("- effR stats -")
    print(f"effR graph has {geffr.numberOfEdges()} edges")
    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data_effr, gsmds.numberOfNodes(), t_orig, t_effr, title=f"ctrl v.s. effR", save_fig=True)
    
    print("- uniform stats -")
    print(f"uniform graph has {gweight.numberOfEdges()} edges")
    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data_uni, gsmds.numberOfNodes(), t_orig, t_uni, title=f"ctrl v.s. uniform", save_fig=True)
    
    print("- weight sampling stats -")
    print(f"weight sampling graph has {gthresh.numberOfEdges()} edges")
    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data_weight, gsmds.numberOfNodes(), t_orig, t_weight, title=f"ctrl v.s. weighted", save_fig=True)
    
    print("- thresholding stats -")
    print(f"thresholding graph has {gsmds.numberOfEdges()} edges")
    sprmn_pval, r_sq, sprmn_rho = spearman_compare_memsafe_diff(data_orig, data_thresh, gsmds.numberOfNodes(), t_orig, t_thresh, title=f"ctrl v.s. thresh", save_fig=True)
    
    print(f"finished after {time.time()-t0} seconds")
    

def viz_graphs(L, sparsity_levels=[0.6], lm = True, multiplicity=False):
    
    print(sparsity_levels)
    increments = [i for i in sparsity_levels]
    t0 = time.time()
    seed = int((time.time_ns()/100)%10000000)
    np.random.seed(seed)
    add_title_data = "sparsities"
    G_init = generate_square_lattice(L, L)
    visualize(G_init)
    
    #target_number_of_nodes = G_init.numberOfNodes() * sparsity
    
    mu_scale = SDRG_crit_point_estimation(G_init, 20)
    
    G_init = scale_mu(G_init, mu_scale)
    
    
    
    sdrg_sparsifications = sdrg_sparsify_partial(G_init, 0.1, False, kawashima=False, local_max=lm, return_incremental_sparsifications=True, increments = increments, use_multiplicity=multiplicity)
    
    
    print(sparsity_levels)
    
    print("Finished all SDRG sparsifications! Moving on to simiulations")
    
    #this data matrix stores everything; each sublist represents the different sparsification techniques at a single sparsity, so:
    """
    for ex
    [
        <90%> [sdrg, smds, effr, ...],
        <80%> [sdrg, smds, effr, ...],
        ...
        ]
    """
    data_matrix = []
    num_edges_matrix = []
    
    for i in range(len(sparsity_levels)):
        sparsity = sparsity_levels[i]
        
        
        G_sdrg = sdrg_sparsifications[i]
        print("- running simulation on sdrg network -")
        visualize(G_sdrg)
        
        
        n_samples = round(sparsity*G_init.numberOfEdges())
        
        print("- running smds for comparison -")
        gsmds = SMDS_to_n_edges(G_init, n_samples)
        visualize(gsmds)
    
        print("- running effR for comparison -")
        geffr = copy_graph(G_init)
        geffr = effective_resistance_sampling_sparsification(geffr, n_samples)
        visualize(geffr)
    
    
        print("- running uniform for comparison -")
        guni = copy_graph(G_init)
        guni = uniform_sampling_sparsification(guni, n_samples) 
        visualize(guni)
        
        print("- running weight sampling for comparison -")
        gweight = copy_graph(G_init)
        gweight = weight_sampling_sparsification(gweight, n_samples) #uniform_sampling_sparsification(guni, n_samples) 
        visualize(gweight)
        
        print("- running thresholding for comparison -")
        gthresh = copy_graph(G_init)
        gthresh =  threshold_sparsification(gthresh, n_samples) #weight_sampling_sparsification(gweight, n_samples) #uniform_sampling_sparsification(guni, n_samples) 
        visualize(gthresh)
        
            
    print(f"finished after {time.time()-t0} seconds")
     

def test_sparsities(L, sparsity_levels=[0.25, 0.4, 0.55, 0.7, 0.85, 0.95], lm = True, multiplicity=False):
    
    print(sparsity_levels)
    
    increments = [i for i in sparsity_levels]
    t0 = time.time()
    seed = int((time.time_ns()/100)%10000000)
    print(seed)
    np.random.seed(seed)
    add_title_data = "sparsities"
    G_init = generate_square_lattice(L, L)
    
    #target_number_of_nodes = G_init.numberOfNodes() * sparsity
    
    mu_scale = SDRG_crit_point_estimation(G_init, 20)
    
    G_init = scale_mu(G_init, mu_scale)
    
    
    spearman_thresh = 0.98
    data_orig, t_orig = fast_dcp_until_quasistationary_memsafe(G_init, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} control {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_orig, f"L{L}x{L}_s{seed}_control_{add_title_data}")
    
    imp_nodes = []#get_nodes_of_largest_k_clusters(G, 3)
    sorted_sites = sorted([data_orig[site][0] for site in data_orig])
    percentile = 0.9
    for site in data_orig:
        #print(ctrl[site])
        if data_orig[site][0] > sorted_sites[round(percentile*len(sorted_sites))]: #t_orig * 0.9:
            imp_nodes.append(site)
    
    
    sdrg_sparsifications = sdrg_sparsify_partial(G_init, 0.1, False, kawashima=False, local_max=lm, return_incremental_sparsifications=True, increments = increments, use_multiplicity=multiplicity)
    
    
    print(sparsity_levels)
    
    print("Finished all SDRG sparsifications! Moving on to simiulations")
    
    #this data matrix stores everything; each sublist represents the different sparsification techniques at a single sparsity, so:
    """
    for ex
    [
        <90%> [sdrg, smds, effr, ...],
        <80%> [sdrg, smds, effr, ...],
        ...
        ]
    """
    data_matrix = []
    num_edges_matrix = []
    
    for i in range(len(sparsity_levels)):
        sparsity = sparsity_levels[i]
        
        
        G_sdrg = sdrg_sparsifications[i]
        print("- running simulation on sdrg network -")
        mu_scale = SDRG_crit_point_estimation(G_sdrg, 20)
        
        G_sdrg = scale_mu(G_sdrg, mu_scale)
        
        data_sdrg, t_sdrg = fast_dcp_until_quasistationary_memsafe(G_sdrg, G_structure="lattice", original_graph_size=G_sdrg.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} sdrgFull {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_sdrg, f"L{L}x{L}_s{seed}_sdrgFull_{add_title_data}_{sparsity}")
        
        n_samples = round(sparsity*G_init.numberOfEdges())
        
        print("- running smds for comparison -")
        gsmds = SMDS_to_n_edges(G_init, n_samples)
        data_smds, t_smds = fast_dcp_until_quasistationary_memsafe(gsmds, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} smds {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_smds, f"L{L}x{L}_s{seed}_smds_{add_title_data}_{sparsity}")
    
        print("- running effR for comparison -")
        geffr = copy_graph(G_init)
        geffr = effective_resistance_sampling_sparsification(geffr, n_samples)
        data_effr, t_effr = fast_dcp_until_quasistationary_memsafe(geffr, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} effr {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_effr, f"L{L}x{L}_s{seed}_effr_{add_title_data}_{sparsity}")
    
        print("- running uniform for comparison -")
        guni = copy_graph(G_init)
        guni = uniform_sampling_sparsification(guni, n_samples) 
        data_uni, t_uni = fast_dcp_until_quasistationary_memsafe(geffr, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} uniform {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_uni, f"L{L}x{L}_s{seed}_uniform_{add_title_data}_{sparsity}")
        
        print("- running weight sampling for comparison -")
        gweight = copy_graph(G_init)
        gweight = weight_sampling_sparsification(gweight, n_samples) #uniform_sampling_sparsification(guni, n_samples) 
        data_weight, t_weight = fast_dcp_until_quasistationary_memsafe(gweight, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} weighted {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_weight, f"L{L}x{L}_s{seed}_weightBased_{add_title_data}_{sparsity}")
        
        print("- running thresholding for comparison -")
        gthresh = copy_graph(G_init)
        gthresh =  threshold_sparsification(gthresh, n_samples) #weight_sampling_sparsification(gweight, n_samples) #uniform_sampling_sparsification(guni, n_samples) 
        data_thresh, t_thresh = fast_dcp_until_quasistationary_memsafe(gthresh, G_structure="lattice", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} thresh {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_thresh, f"L{L}x{L}_s{seed}_thresholding_{add_title_data}_{sparsity}")
    
        data_at_this_sparsity = [data_sdrg, data_smds, data_effr, data_uni, data_weight, data_thresh]
        data_matrix.append(data_at_this_sparsity)
        edge_counts = [G_sdrg.numberOfEdges(), gsmds.numberOfEdges(), geffr.numberOfEdges(), guni.numberOfEdges(), gweight.numberOfEdges(), gthresh.numberOfEdges()]
        num_edges_matrix.append(edge_counts)
        
    print(num_edges_matrix)
    save_data(num_edges_matrix, f"L{L}x{L}_s{seed}_edgeMatrix_{add_title_data}")
    print(seed)
    print("============================")
    print("===== Final Data Stuff =====")
    print("============================")
    
    for i in range(len(sparsity_levels)): 
        sparsity = sparsity_levels[i]
        
        data_at_this_sparsity = data_matrix[i]
        edge_counts = num_edges_matrix[i]
        names = ["SDRG", "SMDS", "effR", "uniform", "weightBased", "thresholding"]
        
        data_sdrg = data_at_this_sparsity[0]
        data_smds = data_at_this_sparsity[1]
        data_effr = data_at_this_sparsity[2]
        data_uni = data_at_this_sparsity[3]
        data_weight = data_at_this_sparsity[4]
        data_thresh = data_at_this_sparsity[5]
        
        print(f"==== {sparsity*100}% of edges remaining data ====")
        
        for j in range(len(names)):
            print(f"- {names[j]} stats -")
            print(f"{names[j]} graph has {edge_counts[j]} edges")
            #sp_pval, r_sq, sp_rho, wass, jens = spearman_compare_memsafe_diff(data_orig, data_at_this_sparsity[j], L*L, data_orig[0][2], data_at_this_sparsity[j][0][2], title=f"{L}x{L} ctrl v.s. {names[j]}: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=True, show_fig=False)
            
            
    print(f"finished after {time.time()-t0} seconds")
    


def test_sparsities_net(N, avg_deg, deg_stdev, G_init = None, sparsity_levels=[0.25, 0.4, 0.55, 0.7, 0.85, 0.95], lm = True, multiplicity=False):
    
    print(sparsity_levels)
    
    increments = [i for i in sparsity_levels]
    t0 = time.time()
    seed = int((time.time_ns()/100)%10000000)
    print(seed)
    np.random.seed(seed)
    add_title_data = "sparsities"
    #G_init = generate_square_lattice(L, L)
    L=N
    if G_init is None:
        G_init = generate_random_graph(N, avg_deg, deg_stdev)
    
        
    #target_number_of_nodes = G_init.numberOfNodes() * sparsity
    
    mu_scale = 1#SDRG_crit_point_estimation(G_init, 20)
    
    G_init = scale_mu(G_init, mu_scale)
    
    
    spearman_thresh = 0.95
    data_orig, t_orig = fast_dcp_until_quasistationary_memsafe(G_init, G_structure="network", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} control {add_title_data}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
    save_data(data_orig, f"L{L}x{L}_s{seed}_control_{add_title_data}")
    
    imp_nodes = []#get_nodes_of_largest_k_clusters(G, 3)
    sorted_sites = sorted([data_orig[site][0] for site in data_orig])
    percentile = 0.9
    for site in data_orig:
        #print(ctrl[site])
        if data_orig[site][0] > sorted_sites[round(percentile*len(sorted_sites))]: #t_orig * 0.9:
            imp_nodes.append(site)
    
    
    sdrg_sparsifications = sdrg_sparsify_partial(G_init, 0.1, False, kawashima=False, local_max=lm, return_incremental_sparsifications=True, increments = increments, use_multiplicity=multiplicity)
    
    
    print(sparsity_levels)
    
    print("Finished all SDRG sparsifications! Moving on to simiulations")
    
    #this data matrix stores everything; each sublist represents the different sparsification techniques at a single sparsity, so:
    """
    for ex
    [
        <90%> [sdrg, smds, effr, ...],
        <80%> [sdrg, smds, effr, ...],
        ...
        ]
    """
    data_matrix = []
    num_edges_matrix = []
    
    for i in range(len(sparsity_levels)):
        sparsity = sparsity_levels[i]
        
        
        G_sdrg = sdrg_sparsifications[i]
        print("- running simulation on sdrg network -")
        mu_scale = SDRG_crit_point_estimation(G_sdrg, 20)
        
        G_sdrg = scale_mu(G_sdrg, mu_scale)
        
        data_sdrg, t_sdrg = fast_dcp_until_quasistationary_memsafe(G_sdrg, G_structure="network", original_graph_size=G_sdrg.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} sdrgFull {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_sdrg, f"L{L}x{L}_s{seed}_sdrgFull_{add_title_data}_{sparsity}")
        
        n_samples = round(sparsity*G_init.numberOfEdges())
        
        print("- running smds for comparison -")
        gsmds = SMDS_to_n_edges(G_init, n_samples)
        data_smds, t_smds = fast_dcp_until_quasistationary_memsafe(gsmds, G_structure="network", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} smds {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_smds, f"L{L}x{L}_s{seed}_smds_{add_title_data}_{sparsity}")
    
        print("- running effR for comparison -")
        geffr = copy_graph(G_init)
        geffr = effective_resistance_sampling_sparsification(geffr, n_samples)
        data_effr, t_effr = fast_dcp_until_quasistationary_memsafe(geffr, G_structure="network", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} effr {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_effr, f"L{L}x{L}_s{seed}_effr_{add_title_data}_{sparsity}")
    
        print("- running uniform for comparison -")
        guni = copy_graph(G_init)
        guni = uniform_sampling_sparsification(guni, n_samples) 
        data_uni, t_uni = fast_dcp_until_quasistationary_memsafe(geffr, G_structure="network", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} uniform {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_uni, f"L{L}x{L}_s{seed}_uniform_{add_title_data}_{sparsity}")
        
        print("- running weight sampling for comparison -")
        gweight = copy_graph(G_init)
        gweight = weight_sampling_sparsification(gweight, n_samples) #uniform_sampling_sparsification(guni, n_samples) 
        data_weight, t_weight = fast_dcp_until_quasistationary_memsafe(gweight, G_structure="network", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} weighted {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_weight, f"L{L}x{L}_s{seed}_weightBased_{add_title_data}_{sparsity}")
        
        print("- running thresholding for comparison -")
        gthresh = copy_graph(G_init)
        gthresh =  threshold_sparsification(gthresh, n_samples) #weight_sampling_sparsification(gweight, n_samples) #uniform_sampling_sparsification(guni, n_samples) 
        data_thresh, t_thresh = fast_dcp_until_quasistationary_memsafe(gthresh, G_structure="network", original_graph_size=G_init.numberOfNodes(), dimensions=[L,L], title=f"{L}x{L} seed={seed} thresh {add_title_data} sparsity={sparsity}", return_time=True, spearman_thresh=spearman_thresh, save_last_state=True, viz=False)
        save_data(data_thresh, f"L{L}x{L}_s{seed}_thresholding_{add_title_data}_{sparsity}")
    
        data_at_this_sparsity = [data_sdrg, data_smds, data_effr, data_uni, data_weight, data_thresh]
        data_matrix.append(data_at_this_sparsity)
        edge_counts = [G_sdrg.numberOfEdges(), gsmds.numberOfEdges(), geffr.numberOfEdges(), guni.numberOfEdges(), gweight.numberOfEdges(), gthresh.numberOfEdges()]
        num_edges_matrix.append(edge_counts)
        
    print(num_edges_matrix)
    save_data(num_edges_matrix, f"L{L}x{L}_s{seed}_edgeMatrix_{add_title_data}")
    print(seed)
    print("============================")
    print("===== Final Data Stuff =====")
    print("============================")
    
    for i in range(len(sparsity_levels)): 
        sparsity = sparsity_levels[i]
        
        data_at_this_sparsity = data_matrix[i]
        edge_counts = num_edges_matrix[i]
        names = ["SDRG", "SMDS", "effR", "uniform", "weightBased", "thresholding"]
        
        data_sdrg = data_at_this_sparsity[0]
        data_smds = data_at_this_sparsity[1]
        data_effr = data_at_this_sparsity[2]
        data_uni = data_at_this_sparsity[3]
        data_weight = data_at_this_sparsity[4]
        data_thresh = data_at_this_sparsity[5]
        
        print(f"==== {sparsity*100}% of edges remaining data ====")
        
        for j in range(len(names)):
            print(f"- {names[j]} stats -")
            print(f"{names[j]} graph has {edge_counts[j]} edges")
            #sp_pval, r_sq, sp_rho, wass, jens = spearman_compare_memsafe_diff(data_orig, data_at_this_sparsity[j], L*L, data_orig[0][2], data_at_this_sparsity[j][0][2], title=f"{L}x{L} ctrl v.s. {names[j]}: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=True, show_fig=False)
            
            
    print(f"finished after {time.time()-t0} seconds")
        
    
    
def extract_data_text_sdrg_smds_compare():
    
    from pathlib import Path

    # Replace with your directory path
    dir_path = Path('C:/Users/zidda/Documents/Summer 26 Research/SDRG_Sparsification_py/sdrg_sparsification/data_files/L16_sdrg_smds_comparison')
    
    # Get all files (ignoring folders)
    files = [f for f in dir_path.iterdir() if f.is_file()]
    #print(files)
    for filename in files:
        filename = str(filename)
        with open(filename, 'r', encoding='utf-8') as file:
            arr = []
            count = 0
            fds_start = 0
            
            flag = False #flag is true when in the "final data stuff" part of the text file
            for line in file:
                line = line.strip()
                #print(line.strip())
                if "===== Final Data Stuff =====" in line:
                    flag = True
                    fds_start = count
                    #print("")
                    #print("------ new trial ------")
                if "done!" in line:
                    flag = False
                #if count - fds_start > 1 and count - fds_start < 18:
                    #print(line)
                if count-fds_start == 9:
                    print(line[22:] + " ", end="")
                
                if count-fds_start == 17:
                    print(line[22:])
                
                
                #if count > 770 and count < 825:
                    #print(line)
                
                count += 1
                
def extract_data_pickles_compare(L, extra_name=""):
    
    from pathlib import Path

    # Replace with your directory path
    #L = 16
    dir_path = Path(f'C:/Users/zidda/Documents/Summer 26 Research/SDRG_Sparsification_py/sdrg_sparsification/data_files/L{L}_pickles{extra_name}')
    
    
    # Get all files (ignoring folders)
    files = [f for f in dir_path.iterdir() if f.is_file()]
    #print(files)
    for file_ind in range(0, len(files)-2, 3):
        ctrl_filename = str(files[file_ind])[:-4]
        seed = ctrl_filename.split("_")[6]
        #ctrl_filename = ctrl_filename[:-4]
        sdrg_filename = str(files[file_ind+1])[:-4]
        smds_filename = str(files[file_ind+2])[:-4]
        #print(ctrl_filename)
        ctrl = open_data(ctrl_filename)
        #print(f"ctrl: {ctrl_filename}")
        sdrg = open_data(sdrg_filename)
        #print(f"sdrg: {sdrg_filename}")
        smds = open_data(smds_filename)
        #if file_ind == 0:
        rough_t_ctrl = ctrl[0][2]
        rough_t_sdrg = sdrg[0][2]
        rough_t_smds = smds[0][2]
        #print(ctrl[0])
        seed_num = int(seed[1:])
        np.random.seed(seed_num)
        #G = generate_square_lattice(L, L)
        
        imp_nodes = []#get_nodes_of_largest_k_clusters(G, 3)
        
        sorted_sites = sorted([ctrl[site][0] for site in ctrl])
        
        for site in ctrl:
            #print(ctrl[site])
            if ctrl[site][0] > sorted_sites[round(0.9*len(sorted_sites))]: #rough_t_ctrl * 0.99:
                imp_nodes.append(site)
                
        if len(imp_nodes) > 1:
            #we'll have tp make a function that gets the nearest higher power of two (i.e, possible max time)
            sp_pval, r_sq, sp_rho_sd, wass_sdrg, jens_sdrg = spearman_compare_memsafe_diff(ctrl, sdrg, L*L, rough_t_ctrl, rough_t_sdrg, title=f"{L}x{L} ctrl v.s. sdrg: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
            sp_pval, r_sq, sp_rho_sm, wass_smds, jens_smds = spearman_compare_memsafe_diff(ctrl, smds, L*L, rough_t_ctrl, rough_t_smds, title=f"{L}x{L} ctrl v.s. smds: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
            
            #vis_clusters(G, L, "lattice", f"{L}x{L} clusters: seed={seed_num}")
            
            #vis_lat_advanced_memsafe(ctrl, L*L, rough_t_ctrl, "lattice", [L, L], f"{L}x{L} ctrl: seed={seed_num}", nodes_to_highlight=imp_nodes)
            #vis_lat_advanced_memsafe(sdrg, L*L, rough_t_sdrg, "lattice", [L, L], f"{L}x{L} sdrg: seed={seed_num}", nodes_to_highlight=imp_nodes)
            #vis_lat_advanced_memsafe(smds, L*L, rough_t_smds, "lattice", [L, L], f"{L}x{L} smds: seed={seed_num}", nodes_to_highlight=imp_nodes)
            #print(f"-- {seed_num} --")
            #print(f"{rough_t_ctrl}\t{rough_t_sdrg}\t{rough_t_smds}")
            #print(f"{sp_rho_sd}\t{sp_rho_sm}")
            print(f"{jens_sdrg}\t{jens_smds}")
            #print(f"{wass_sdrg}\t{wass_smds}")
          
def extract_data_pickles_compare_4(L, extra_name=""):
    
    from pathlib import Path

    # Replace with your directory path
    #L = 16
    dir_path = Path(f'C:/Users/zidda/Documents/Summer 26 Research/SDRG_Sparsification_py/sdrg_sparsification/data_files/L{L}_pickles{extra_name}')
    
    
    # Get all files (ignoring folders)
    files = [f for f in dir_path.iterdir() if f.is_file()]
    #print(files)
    
    sdrg_data = []
    smds_data = []
    st_data = []
    bins = np.linspace(0,1.5, 31)#range(0,1.5,0.05)
    
    for file_ind in range(0, len(files)-2, 4):
        ctrl_filename = str(files[file_ind])[:-4]
        seed = ctrl_filename.split("_")[7]
        #ctrl_filename = ctrl_filename[:-4]
        sdrg_filename = str(files[file_ind+1])[:-4]
        smds_filename = str(files[file_ind+2])[:-4]
        st_filename = str(files[file_ind+3])[:-4]
        #print(ctrl_filename)
        ctrl = open_data(ctrl_filename)
        #print(f"ctrl: {ctrl_filename}")
        sdrg = open_data(sdrg_filename)
        #print(f"sdrg: {sdrg_filename}")
        smds = open_data(smds_filename)
        
        st = open_data(st_filename)
        #if file_ind == 0:
        rough_t_ctrl = ctrl[0][2]
        rough_t_sdrg = sdrg[0][2]
        rough_t_smds = smds[0][2]
        rough_t_st = st[0][2]
        #print(ctrl[0])
        seed_num = int(seed[1:])
        np.random.seed(seed_num)
        #G = generate_square_lattice(L, L)
        
        imp_nodes = []#get_nodes_of_largest_k_clusters(G, 3)
        sorted_sites = sorted([ctrl[site][0] for site in ctrl])
        
        #print(sorted_sites[round(0.98*len(sorted_sites))] / rough_t_ctrl)
        for site in ctrl:
            #print(ctrl[site])
            #if a site is in the top 10%
            if ctrl[site][0] > sorted_sites[round(0.9*len(sorted_sites))]: #rough_t_ctrl * 0.5:
                imp_nodes.append(site)
        
        
        #add in any others
        if len(imp_nodes) > 1:
            #we'll have tp make a function that gets the nearest higher power of two (i.e, possible max time)
            sp_pval, r_sq, sp_rho_sd, wass_sdrg, jens_sdrg = spearman_compare_memsafe_diff(ctrl, sdrg, L*L, rough_t_ctrl, rough_t_sdrg, title=f"{L}x{L} ctrl v.s. sdrg: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
            sp_pval, r_sq, sp_rho_st, wass_st, jens_st = spearman_compare_memsafe_diff(ctrl, st, L*L, rough_t_ctrl, rough_t_st, title=f"{L}x{L} ctrl v.s. st: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
            sp_pval, r_sq, sp_rho_sm, wass_smds, jens_smds = spearman_compare_memsafe_diff(ctrl, smds, L*L, rough_t_ctrl, rough_t_smds, title=f"{L}x{L} ctrl v.s. smds: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
            
            #vis_clusters(G, L, "lattice", f"{L}x{L} clusters: seed={seed_num}")
            
            #vis_lat_advanced_memsafe(ctrl, L*L, rough_t_ctrl, "lattice", [L, L], f"{L}x{L} ctrl: seed={seed_num}", nodes_to_highlight=imp_nodes)
            #vis_lat_advanced_memsafe(sdrg, L*L, rough_t_sdrg, "lattice", [L, L], f"{L}x{L} sdrg: seed={seed_num}", nodes_to_highlight=imp_nodes)
            #vis_lat_advanced_memsafe(smds, L*L, rough_t_smds, "lattice", [L, L], f"{L}x{L} smds: seed={seed_num}", nodes_to_highlight=imp_nodes)
            #print(f"-- {seed_num} --")
            #print(f"{rough_t_ctrl}\t{rough_t_sdrg}\t{rough_t_smds}")
            #print(f"{sp_rho_sd}\t{sp_rho_sm}")
            print(f"{jens_sdrg}\t{jens_st}\t{jens_smds}")
            sdrg_data.append(jens_sdrg)
            smds_data.append(jens_smds)
            st_data.append(jens_st)
            #print(f"{wass_sdrg}\t{wass_smds}")
        
    
    #print(sdrg_data)
    sns.histplot(sdrg_data, bins=bins, color="blue", label="SDRG", alpha=0.4, element="step")
    sns.histplot(smds_data, bins=bins, color="orange", label="SMDS", alpha=0.4, element="step")
    
    # 4. Finalize the chart details
    plt.xlabel("Wasserstein Distance - Most Active 10%")
    plt.ylabel("Frequency")
    plt.title("SDRG v.s. SMDS Quasistationary Wasserstein Distance from Control")
    plt.legend(loc="upper right")
    
    plt.show()

def extract_data_pickles_compare_all(L, num_tests, extra_name=""):
    
    from pathlib import Path

    # Replace with your directory path
    #L = 16
    dir_path = Path(f'C:/Users/zidda/Documents/Summer 26 Research/SDRG_Sparsification_py/sdrg_sparsification/data_files/L{L}_pickles{extra_name}')
    
    
    # Get all files (ignoring folders)
    files = [f for f in dir_path.iterdir() if f.is_file()]
    #print(files)
    
    sdrg_data = []
    smds_data = []
    st_data = []
    
    test_datas = [[], [], [], [], [], [], []]#seven empties, one for each test
    
    #this controls the percentile of activity we examine. 0.0 gives all nodes, 0.95 gives the top 5% by activity, etc.
    percentile = 0.9
    
    bins = np.linspace(0,2, 41)#range(0,1.5,0.05)
    #num_tests = 7
    names = ["ctrl", "effr", "sdrg", "smds", "thresh", "uniform", "weight"]
    colors = ["white", "red", "orange", "yellow", "green", "blue", "purple", "black"]
    for file_ind in range(0, len(files)-1, num_tests):
        ctrl_filename = str(files[file_ind])[:-4]
        ctrl = open_data(ctrl_filename)
        seed = ctrl_filename.split("_")[7] #should be 7
        #"C:\Users\zidda\Documents\Summer 26 Research\SDRG_Sparsification_py\sdrg_sparsification\data_files\L16_pickles_alltests\L16x16_s4239702_control_st_sdrg.pkl"
        seed_num = int(seed[1:])
        np.random.seed(seed_num)
        
        imp_nodes = []#get_nodes_of_largest_k_clusters(G, 3)
        sorted_sites = sorted([ctrl[site][0] for site in ctrl])
        
        #print(sorted_sites[round(0.98*len(sorted_sites))] / rough_t_ctrl)
        for site in ctrl:
            #print(ctrl[site])
            #if a site is in the top 10%
            if ctrl[site][0] > 0:#sorted_sites[round(percentile*len(sorted_sites))]: #rough_t_ctrl * 0.5:
                imp_nodes.append(site)
        
        rough_t_ctrl = ctrl[0][2]
        
        for i in range(1, num_tests):
            filename = str(files[file_ind+i])[:-4]
            test_data = open_data(filename)
            rough_t = test_data[0][2]
            test_name = filename.split("_")[8] #should be 8
            #print(test_name)
            #print(test_data[0][0])
        
            #add in any others
            if len(imp_nodes) > 1:
                #we'll have tp make a function that gets the nearest higher power of two (i.e, possible max time)
                try:
                    sp_pval, r_sq, sp_rho, wass, jens = spearman_compare_memsafe_diff(ctrl, test_data, L*L, rough_t_ctrl, rough_t, title=f"{L}x{L} ctrl v.s. {test_name}: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
                except:
                    pass    
                #sp_pval, r_sq, sp_rho_st, wass_st, jens_st = spearman_compare_memsafe_diff(ctrl, st, L*L, rough_t_ctrl, rough_t_st, title=f"{L}x{L} ctrl v.s. st: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
                #sp_pval, r_sq, sp_rho_sm, wass_smds, jens_smds = spearman_compare_memsafe_diff(ctrl, smds, L*L, rough_t_ctrl, rough_t_smds, title=f"{L}x{L} ctrl v.s. smds: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
                
                #vis_clusters(G, L, "lattice", f"{L}x{L} clusters: seed={seed_num}")
                
                #vis_lat_advanced_memsafe(ctrl, L*L, rough_t_ctrl, "lattice", [L, L], f"{L}x{L} ctrl: seed={seed_num}", nodes_to_highlight=imp_nodes)
                #vis_lat_advanced_memsafe(sdrg, L*L, rough_t_sdrg, "lattice", [L, L], f"{L}x{L} sdrg: seed={seed_num}", nodes_to_highlight=imp_nodes)
                #vis_lat_advanced_memsafe(smds, L*L, rough_t_smds, "lattice", [L, L], f"{L}x{L} smds: seed={seed_num}", nodes_to_highlight=imp_nodes)
                #print(f"-- {seed_num} --")
                #print(f"{rough_t_ctrl}\t{rough_t_sdrg}\t{rough_t_smds}")
                #print(f"{sp_rho_sd}\t{sp_rho_sm}")
                #print(f"{round(jens, 3)}", end="\t")
                
                if jens == 0:
                    jens = 100
                    print(test_name)
                
                test_datas[i].append(jens)
                #sdrg_data.append(jens_sdrg)
                #smds_data.append(jens_smds)
                #st_data.append(jens_st)
                #print(f"{wass_sdrg}\t{wass_smds}")
        print("--")
        
    
    #print(sdrg_data)
    for i in range(num_tests):
        sns.histplot(test_datas[i], bins=bins, color=colors[i], label=names[i], alpha=0.4, element="step")
    #sns.histplot(smds_data, bins=bins, color="orange", label="SMDS", alpha=0.4, element="step")
    
    # 4. Finalize the chart details
    plt.xlabel(f"Wasserstein Distance - Most Active {(1-percentile)*100}%")
    plt.ylabel("Frequency")
    plt.title("Quasistationary Wasserstein Distance from Control")
    plt.legend(loc="upper right")
    
    plt.show()
         

def extract_data_pickles_compare_all_sparsities(L, num_tests, extra_name=""):
    
    from pathlib import Path

    # Replace with your directory path
    #L = 16
    dir_path = Path(f'C:/Users/zidda/Documents/Summer 26 Research/SDRG_Sparsification_py/sdrg_sparsification/data_files/L{L}_pickles{extra_name}')
    
    
    # Get all files (ignoring folders)
    files = [f for f in dir_path.iterdir() if f.is_file()]
    #print(files)
    
    sdrg_data = []
    smds_data = []
    st_data = []
    
    test_datas = [[], [], [], [], [], [], []]#seven empties, one for each test
    
    #this controls the percentile of activity we examine. 0.0 gives all nodes, 0.95 gives the top 5% by activity, etc.
    percentile = 0.9
    
    bins = np.linspace(0,2, 41)#range(0,1.5,0.05)
    #num_tests = 7
    names = ["ctrl", "effr", "sdrg", "smds", "thresh", "uniform", "weight"]
    colors = ["white", "red", "orange", "yellow", "green", "blue", "purple", "black"]
    for file_ind in range(0, len(files)-1, 50):
        ctrl_filename = str(files[file_ind])[:-4]
        ctrl = open_data(ctrl_filename)
        seed = ctrl_filename.split("_")[7] #should be 7
        #print(seed)
        #"C:\Users\zidda\Documents\Summer 26 Research\SDRG_Sparsification_py\sdrg_sparsification\data_files\L16_pickles_alltests\L16x16_s4239702_control_st_sdrg.pkl"
        seed_num = int(seed[1:])
        np.random.seed(seed_num)
        
        imp_nodes = []#get_nodes_of_largest_k_clusters(G, 3)
        sorted_sites = sorted([ctrl[site][0] for site in ctrl])
        
        #print(sorted_sites[round(0.98*len(sorted_sites))] / rough_t_ctrl)
        for site in ctrl:
            #print(ctrl[site])
            #if a site is in the top 10%
            if ctrl[site][0] > 0:#sorted_sites[round(percentile*len(sorted_sites))]: #rough_t_ctrl * 0.5:
                imp_nodes.append(site)
        
        rough_t_ctrl = ctrl[0][2]
        edge_mat_filename = str(files[file_ind+1])[:-4]
        edge_matrix = open_data(edge_mat_filename)
        #edgematrix has: sdrg, smds, effr, uniform, weight, thresh
        #sparsity_ind = 3
        """for sparsity_ind in range(9):
            print(edge_matrix[sparsity_ind][2], end="\t")
            print(edge_matrix[sparsity_ind][0], end="\t")
            print(edge_matrix[sparsity_ind][1], end="\t")
            print(edge_matrix[sparsity_ind][5], end="\t")
            print(edge_matrix[sparsity_ind][3], end="\t")
            print(edge_matrix[sparsity_ind][4], end="\t")
            print("\t\t\t\t\t\t\t\t", end="")"""
        
        for i in range(4, 50, 8):
            filename = str(files[file_ind+i])[:-4]
            test_data = open_data(filename)
            rough_t = test_data[0][2]
            test_name = filename.split("_")[8] #should be 8
            #print(test_name, end="-")
            #print(test_name)
            #print(test_data[0][0])
        
            #add in any others
            
            sp_pval, r_sq, sp_rho, wass, jens = spearman_compare_memsafe_diff(ctrl, test_data, L*L, rough_t_ctrl, rough_t, title=f"{L}x{L} ctrl v.s. {test_name}: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
            print(f"{jens}", end="\t")
            #sp_pval, r_sq, sp_rho_st, wass_st, jens_st = spearman_compare_memsafe_diff(ctrl, st, L*L, rough_t_ctrl, rough_t_st, title=f"{L}x{L} ctrl v.s. st: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
            #sp_pval, r_sq, sp_rho_sm, wass_smds, jens_smds = spearman_compare_memsafe_diff(ctrl, smds, L*L, rough_t_ctrl, rough_t_smds, title=f"{L}x{L} ctrl v.s. smds: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
            
            #vis_clusters(G, L, "lattice", f"{L}x{L} clusters: seed={seed_num}")
            
            #vis_lat_advanced_memsafe(ctrl, L*L, rough_t_ctrl, "lattice", [L, L], f"{L}x{L} ctrl: seed={seed_num}", nodes_to_highlight=imp_nodes)
            #vis_lat_advanced_memsafe(sdrg, L*L, rough_t_sdrg, "lattice", [L, L], f"{L}x{L} sdrg: seed={seed_num}", nodes_to_highlight=imp_nodes)
            #vis_lat_advanced_memsafe(smds, L*L, rough_t_smds, "lattice", [L, L], f"{L}x{L} smds: seed={seed_num}", nodes_to_highlight=imp_nodes)
            #print(f"-- {seed_num} --")
            #print(f"{rough_t_ctrl}\t{rough_t_sdrg}\t{rough_t_smds}")
            #print(f"{sp_rho_sd}\t{sp_rho_sm}")
            #print(f"{round(jens, 3)}", end="\t")
            
            if jens == 0:
                jens = 100
                print(test_name)
            
            #test_datas[i].append(jens)
            #sdrg_data.append(jens_sdrg)
            #smds_data.append(jens_smds)
            #st_data.append(jens_st)
            #print(f"{wass_sdrg}\t{wass_smds}")
        print("")
        
    
    #print(sdrg_data)
    for i in range(num_tests):
        sns.histplot(test_datas[i], bins=bins, color=colors[i], label=names[i], alpha=0.4, element="step")
    #sns.histplot(smds_data, bins=bins, color="orange", label="SMDS", alpha=0.4, element="step")
    
    # 4. Finalize the chart details
    plt.xlabel(f"Wasserstein Distance - Most Active {(1-percentile)*100}%")
    plt.ylabel("Frequency")
    plt.title("Quasistationary Wasserstein Distance from Control")
    plt.legend(loc="upper right")
    
    plt.show()
         

def extract_data_pickles_compare_all_sparsities_increments(L, num_tests, extra_name=""):
    
    from pathlib import Path
    increments = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    already_done = 0
    for inc in range(2+already_done, len(increments)+2):
        print("=================================")
        print(f"=============={increments[inc-2]}==============")
        print("=================================")
        # Replace with your directory path
        #L = 16
        dir_path = Path(f'C:/Users/zidda/Documents/Summer 26 Research/SDRG_Sparsification_py/sdrg_sparsification/data_files/L{L}_pickles{extra_name}')
        
        
        # Get all files (ignoring folders)
        files = [f for f in dir_path.iterdir() if f.is_file()]
        #print(files)
        
        sdrg_data = []
        smds_data = []
        st_data = []
        
        test_datas = [[], [], [], [], [], [], []]#seven empties, one for each test
        
        #this controls the percentile of activity we examine. 0.0 gives all nodes, 0.95 gives the top 5% by activity, etc.
        percentile = 0.95
        
        bins = np.linspace(0,2, 41)#range(0,1.5,0.05)
        #num_tests = 7
        names = ["ctrl", "effr", "sdrg", "smds", "thresh", "uniform", "weight"]
        colors = ["white", "red", "orange", "yellow", "green", "blue", "purple", "black"]
        for file_ind in range(0, len(files)-1, 50):
            ctrl_filename = str(files[file_ind])[:-4]
            ctrl = open_data(ctrl_filename)
            seed = ctrl_filename.split("_")[7] #should be 7
            #print(seed)
            #"C:\Users\zidda\Documents\Summer 26 Research\SDRG_Sparsification_py\sdrg_sparsification\data_files\L16_pickles_alltests\L16x16_s4239702_control_st_sdrg.pkl"
            seed_num = int(seed[1:])
            np.random.seed(seed_num)
            
            imp_nodes = []#get_nodes_of_largest_k_clusters(G, 3)
            sorted_sites = sorted([ctrl[site][0] for site in ctrl])
            
            #print(sorted_sites[round(0.98*len(sorted_sites))] / rough_t_ctrl)
            for site in ctrl:
                #print(ctrl[site])
                #if a site is in the top 10%
                if ctrl[site][0] > 0:#sorted_sites[round(percentile*len(sorted_sites))]: #rough_t_ctrl * 0.5:
                    imp_nodes.append(site)
            
            rough_t_ctrl = ctrl[0][2]
            edge_mat_filename = str(files[file_ind+1])[:-4]
            edge_matrix = open_data(edge_mat_filename)
            #edgematrix has: sdrg, smds, effr, uniform, weight, thresh
            #sparsity_ind = 3
            """for sparsity_ind in range(9):
                print(edge_matrix[sparsity_ind][2], end="\t")
                print(edge_matrix[sparsity_ind][0], end="\t")
                print(edge_matrix[sparsity_ind][1], end="\t")
                print(edge_matrix[sparsity_ind][5], end="\t")
                print(edge_matrix[sparsity_ind][3], end="\t")
                print(edge_matrix[sparsity_ind][4], end="\t")
                print("\t\t\t\t\t\t\t\t", end="")"""
            
            for i in range(inc, 50, 8):
                filename = str(files[file_ind+i])[:-4]
                test_data = open_data(filename)
                rough_t = test_data[0][2]
                test_name = filename.split("_")[8] #should be 8
                #print(test_name, end="-")
                #print(test_name)
                #print(test_data[0][0])
            
                #add in any others
                
                sp_pval, r_sq, sp_rho, wass, jens = spearman_compare_memsafe_diff(ctrl, test_data, L*L, rough_t_ctrl, rough_t, title=f"{L}x{L} ctrl v.s. {test_name}: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
                print(f"{jens}", end="\t")
                #sp_pval, r_sq, sp_rho_st, wass_st, jens_st = spearman_compare_memsafe_diff(ctrl, st, L*L, rough_t_ctrl, rough_t_st, title=f"{L}x{L} ctrl v.s. st: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
                #sp_pval, r_sq, sp_rho_sm, wass_smds, jens_smds = spearman_compare_memsafe_diff(ctrl, smds, L*L, rough_t_ctrl, rough_t_smds, title=f"{L}x{L} ctrl v.s. smds: seed={seed}", wass_out = True, jensen_out = True, important_nodes = imp_nodes, verbose=False, show_fig=False)
                
                #vis_clusters(G, L, "lattice", f"{L}x{L} clusters: seed={seed_num}")
                
                #vis_lat_advanced_memsafe(ctrl, L*L, rough_t_ctrl, "lattice", [L, L], f"{L}x{L} ctrl: seed={seed_num}", nodes_to_highlight=imp_nodes)
                #vis_lat_advanced_memsafe(sdrg, L*L, rough_t_sdrg, "lattice", [L, L], f"{L}x{L} sdrg: seed={seed_num}", nodes_to_highlight=imp_nodes)
                #vis_lat_advanced_memsafe(smds, L*L, rough_t_smds, "lattice", [L, L], f"{L}x{L} smds: seed={seed_num}", nodes_to_highlight=imp_nodes)
                #print(f"-- {seed_num} --")
                #print(f"{rough_t_ctrl}\t{rough_t_sdrg}\t{rough_t_smds}")
                #print(f"{sp_rho_sd}\t{sp_rho_sm}")
                #print(f"{round(jens, 3)}", end="\t")
                
                if jens == 0:
                    jens = 100
                    print(test_name)
                
                #test_datas[i].append(jens)
                #sdrg_data.append(jens_sdrg)
                #smds_data.append(jens_smds)
                #st_data.append(jens_st)
                #print(f"{wass_sdrg}\t{wass_smds}")
            print("")
            
        
        #print(sdrg_data)
        for i in range(num_tests):
            sns.histplot(test_datas[i], bins=bins, color=colors[i], label=names[i], alpha=0.4, element="step")
        #sns.histplot(smds_data, bins=bins, color="orange", label="SMDS", alpha=0.4, element="step")
        
        # 4. Finalize the chart details
        plt.xlabel(f"Wasserstein Distance - Most Active {(1-percentile)*100}%")
        plt.ylabel("Frequency")
        plt.title("Quasistationary Wasserstein Distance from Control")
        plt.legend(loc="upper right")
        
        plt.show()

#C:\Users\zidda\Documents\Summer 26 Research\SDRG_Sparsification_py\sdrg_sparsification\data_files\L32_pickles\L32x32_s12806_control_backbone_test.pkl

#test11(4, 1)
#test_sparsities(32)
#for i in range(3):
    #print(f"starting test number {i}")
    #print(f"{time.localtime()[3]}hrs : {time.localtime()[4]}m : {time.localtime()[5]}s")
    #compare_min_partial_sparsifications(16)
    
