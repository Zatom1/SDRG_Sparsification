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
import random
import bisect
np.random.seed(1)
    #"C:\Users\zidda\Downloads\inputs_L10_pbcTrue_h0-1.0_j0-1.0_seed1.npz"
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
    
    
def sdrg_step(G, neil_mode = False, decimated_sites=[], logging_toggle = True, decimation_log = [], visualizeStep = False):
    #Does a single iteration of the SDRG sparsification
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
        # sets automatically remove the duplicates
        pair_neighborhood = {u for u in G.iterNeighbors(edge_to_decimate[0])} | {v for v in G.iterNeighbors(edge_to_decimate[1])}
        #neighbors_checked = set(())
        u = edge_to_decimate[0]
        v = edge_to_decimate[1]
        #print(f"decimating edge {edge_to_decimate} with lambda = {G.weight(u,v)}")

        k = G.addNode() # returns new node id, so k = new node id
        if logging_toggle:
            decimation_log.append([len(decimation_log), edge_to_decimate, G.weight(u,v)])
        
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
            G.addEdge(k, neighbor, new_edge_weight)
            #np.append(lambda_arr, np.array([new_edge_weight, k, neighbor]).T)
            
        #remove nodes at end. We had to wait b/c otherwise we can't calculate weights in loop
        G.removeNode(u)
        G.removeNode(v)
        
    else: #true iff biggest val is a mu
        
        #since we put the mu_arr in front of the lambda_arr, we can directly access the mu_arr list with our max_mu_lambda_index without getting indexing errors
        site_to_decimate = mu_arr[max_mu_lambda_index,1]
        #print(f"decimating site {site_to_decimate} with mu = {healing_factor[site_to_decimate]}")
        #build a set with all of 
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
                    #r_hi = -math.log(healing_factor[site_to_decimate])
                    #kappa_ik = -math.log(G.weight(site_to_decimate, other_neighbor))
                    #kappa_ij = -math.log(G.weight(neighbor, site_to_decimate))
                    
                    a = (G.weight(site_to_decimate, other_neighbor)*G.weight(neighbor, site_to_decimate))/healing_factor[site_to_decimate]
                    
                    new_weight = max(J_jk, a)
                    
                    #G.removeEdge(neighbor, other_neighbor)
                    add_success = G.addEdge(neighbor, other_neighbor, new_weight, checkMultiEdge = True)
                    if not add_success:
                        G.setWeight(neighbor, other_neighbor, new_weight)
                    #G.setWeight(neighbor, other_neighbor, new_weight)
                    
            #The neighbor that just looped through all of the other neighbors will
            # have had all of its connections made and calculated, so no reason to 
            # do anything to it for the rest of the loop
            neighbors_checked.add(neighbor)
            
        decimated_sites = np.append(decimated_sites, components[site_to_decimate])
        #print(f"decimated site {site_to_decimate} with components {components[site_to_decimate]}")
        if logging_toggle:
            decimation_log.append([len(decimation_log), site_to_decimate, healing_factor[site_to_decimate]])

        G.removeNode(site_to_decimate)
        
    #nk.graphio.writeGraph(G, f"network_{time.time_ns()}_T.gml", nk.Format.GML)
    if visualizeStep:
        visualize(G)
    if logging_toggle:
        return G, decimation_log
    elif neil_mode:
        return G, decimated_sites
    return G


#This might be obsolete;
def sdrg_step_edge(G, neil_mode = False, decimated_sites=[], logging_toggle = True, decimation_log = [], visualizeStep = False):
    n_nodes = G.numberOfNodes()
    healing_factor = G.getNodeAttribute("mu", float)
    components = G.getNodeAttribute("components", str)
    is_active = G.getNodeAttribute("active", int)
    if n_nodes == 1:
        print("Network is fully sparsified")
        last_site = [u for u in G.iterNodes()]

        decimated_sites = np.append(decimated_sites, components[last_site[0]])
        if logging_toggle:
            return G, decimation_log
        elif neil_mode:
            return G, decimated_sites
        return G
    
    mu_arr = np.array([np.array([healing_factor[u], u]).T for u in G.iterNodes()])
    lambda_arr = np.array([np.array([G.weight(u,v), u, v]).T for u,v in G.iterEdges()])
    
    lambda_arr = lambda_arr*0
    #print(lambda_arr[:,0])
        
    max_mu_lambda_index = np.argmax(np.concat((mu_arr[:,0], lambda_arr[:,0])))
    #print(max_mu_lambda_index)
    if max_mu_lambda_index > n_nodes-1: #true iff biggest value is a lambda
        edge_to_decimate = lambda_arr[max_mu_lambda_index-n_nodes, 1:] #np list of length 2
        
        #union the two sets containing neighbors of u and v
        # sets automatically remove the duplicates
        pair_neighborhood = {u for u in G.iterNeighbors(edge_to_decimate[0])} | {v for v in G.iterNeighbors(edge_to_decimate[1])}
        #neighbors_checked = set(())
        u = edge_to_decimate[0]
        v = edge_to_decimate[1]
        #print(f"decimating edge {edge_to_decimate} with lambda = {G.weight(u,v)}")

        k = G.addNode() # returns new node id, so k = new node id
        if logging_toggle:
            decimation_log.append([len(decimation_log), edge_to_decimate, G.weight(u,v)])
        
        #use log and exp to convert mults and divides to adds and substracts
        #math.exp(math.log(healing_factor[u]) + math.log(healing_factor[v]) - math.log(G.weight(u,v)))
        h_k = (healing_factor[u]*healing_factor[v])/(G.weight(u,v))
        #np.append(mu_arr, h_k)
        
        healing_factor[k] = h_k
        components[k] = f"{components[u]}_{components[v]}"
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
        
        #print(f"decimated edge between nodes {u} and {v} to create node {k} with components {components[k]}")
        
        
        #print(edge_to_decimate)
        
        
        
    #nk.graphio.writeGraph(G, f"network_{time.time_ns()}_T.gml", nk.Format.GML)
    if visualizeStep:
        visualize(G)
    if logging_toggle:
        return G, decimation_log
    elif neil_mode:
        return G, decimated_sites
    return G




def get_neil_output(G):
    decimated_sites = []
    for i in range(G.numberOfNodes()):
        G, decimated_sites = sdrg_step(G, neil_mode = False, decimated_sites=decimated_sites, visualizeStep=False)
        
    #print(decimated_sites)
    print("--------------------------")

    temp = [components.split("_") for components in decimated_sites]
    formatted_decimated_sites = [sorted([int(x) for x in sublist]) for sublist in temp]
    print(formatted_decimated_sites)
    


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


def prop_sdrg(G, proportion, visualizeSteps = False):
    """
    This function keeps <proportion> percent of the network's nodes
    """
    n = G.numberOfNodes()-round(G.numberOfNodes()*proportion)
    decimated_sites = []
    for i in range(n):
        G, decimated_sites = sdrg_step(G, decimated_sites, visualizeStep=visualizeSteps)
        
    #print(decimated_sites)
    print("--------------------------")
        
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
        
def visualize(G, active_nodes=[], node_size = 100):
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
        if step_counter%1000 == 0:
            #pass
            print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker


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
        if step_counter%1000 == 0:
            #pass
            print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps. The last timestep was {deltaT}")
    if track_TOA:
        return TOA_tracker
    else:
        return df

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

"""
We want to profile the network based on a few characteristics:
    - Rate of infection spread / Time Of Arrival accuracy
    - Check if sparsified networks have same lambda_critical
    - etc.?
The following functions handle these characterizations

"""

def swap_column_vals_after_k(df, row, k):
    for i in range(k, df.shape[1]):
        if df.iat[row, i] == 0:
            df.iat[row, i] = 1
        else:
            df.iat[row, i] = 0
    return df

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
    return i-1
    #raise ValueError

def asymptotic_quasistationary_activity_probability(G, G_structure="chain"):
    relaxation_time = 10000
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


def sparsified_asymptotic_quasistationary_activity_probability(G, G_structure="chain", original_graph_size=100):
    relaxation_time = 10000
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

        Z = np.random.rand(6, 10)
        x = np.arange(-0.5, 10, 1)  # len = 11
        y = np.arange(4.5, 11, 1)  # len = 7
        
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, Z)
        
        fig.tight_layout()
        plt.show()
    
    #data.to_excel("output_{time.time_ns()}.xlsx")
    return data


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



"""

- Check weight based, effR, semi-metric, etc
- Use quasistationary simulation (don't let infection ever die out)
- how did the clusters get determined for DCP from the 2020 paper?
    - which metrics are being tracked in the 2020 paper from the DCP there?
"""
#x = sdrg(*generate_graph(500, 50, 20, 0.1, 0.01))

