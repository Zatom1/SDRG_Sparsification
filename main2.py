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

np.random.seed(1)
    
def translate_Neil_NP_to_graph(filename='inputs_L10_pbcTrue_h0-1.0_j0-1.0_seed1.npz'):
    
    with np.load(filename) as data:
        print("Keys in file:", data.files)
        L = data[data.files[0]]
        r = data[data.files[1]]
        kappa = data[data.files[2]]
        bonds = data[data.files[3]]
        n_sites = data[data.files[4]]
        n_bonds = data[data.files[5]]
        
        G = nk.graph.Graph(n=n_sites, weighted = True, edgesIndexed = True)
        
        #init activity values (infected)
        is_active = G.attachNodeAttribute("active", int)
        
        #init mu values
        healing_factor = G.attachNodeAttribute("mu", float)
        components = G.attachNodeAttribute("components", str)
        
        for i in range(len(bonds)):
            G.addEdge(bonds[i,0], bonds[i,1])
            J = math.exp(-1*kappa[i])
            G.setWeight(bonds[i,0], bonds[i,1], J)
            #print(f"J = {J}")
        
        for u in G.iterNodes():
            is_active[u] = False
            h = math.exp(-1*r[u])
            healing_factor[u] = h
            components[u] = f"{u}"
            #print(f"h = {h}")
        
        visualize(G)
        return G
        
        print(bonds)
def generate_square_lattice(width, height, torus = True, visualize_on=False, disordered_mu = False, constant_mu = 1.0, disordered_lambda=False, constant_lambda = 0.8):
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
    
    
def sdrg_step(G, decimated_sites = [], visualizeStep = False):
    n_nodes = G.numberOfNodes()
    healing_factor = G.getNodeAttribute("mu", float)
    components = G.getNodeAttribute("components", str)
    is_active = G.getNodeAttribute("active", int)
    if n_nodes == 1:
        print("Network is fully sparsified")
        last_site = [u for u in G.iterNodes()]
        print(last_site)
        decimated_sites = np.append(decimated_sites, components[last_site[0]])
        return G, decimated_sites
    
    mu_arr = np.array([np.array([healing_factor[u], u]).T for u in G.iterNodes()])
    lambda_arr = np.array([np.array([G.weight(u,v), u, v]).T for u,v in G.iterEdges()])
    
    
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
        
        
        #use log and exp to convert mults and divides to adds and substracts
        h_k = (healing_factor[u]*healing_factor[v])/(G.weight(u,v))#math.exp(math.log(healing_factor[u]) + math.log(healing_factor[v]) - math.log(G.weight(u,v)))
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
        
        print(f"decimated edge between nodes {u} and {v} to create node {k} with components {components[k]}")
        
        
        #print(edge_to_decimate)
        
        
    else: #true iff biggest val is a mu
        
        site_to_decimate = mu_arr[max_mu_lambda_index,1]
        #print(f"decimating site {site_to_decimate} with mu = {healing_factor[site_to_decimate]}")
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
                    
                    a = (G.weight(site_to_decimate, other_neighbor)*G.weight(neighbor, site_to_decimate))/healing_factor[site_to_decimate]
                    
                    new_weight = max(J_jk, a)#math.exp(kappa_ik + kappa_ij - r_hi))
                    
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
        decimated_sites = np.append(decimated_sites, components[site_to_decimate])
        print(f"decimated site {site_to_decimate} with components {components[site_to_decimate]}")
        G.removeNode(site_to_decimate)
        
        
        
        
    #nk.graphio.writeGraph(G, f"network_{time.time_ns()}_T.gml", nk.Format.GML)
    if visualizeStep:
        visualize(G)
    return G, decimated_sites

def get_neil_output(G):
    decimated_sites = []
    for i in range(G.numberOfNodes()):
        G, decimated_sites = sdrg_step(G, decimated_sites, visualizeStep=False)
        
    #print(decimated_sites)
    print("--------------------------")

    temp = [components.split("_") for components in decimated_sites]
    formatted_decimated_sites = [sorted([int(x) for x in sublist]) for sublist in temp]
    print(formatted_decimated_sites)
    


def full_sdrg(G, visualizeSteps = False):
    G_mod = G
    n_nodes = G.numberOfNodes()
    while n_nodes > 1:
        G_mod = sdrg_step(G_mod, visualizeStep = visualizeSteps)
        n_nodes = G_mod.numberOfNodes()
        
    return G_mod

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
    
    
def DCP_v0(G, t_max = 1, random_start=True, lattice = True, torus = True):
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
    I'm implementing the SIS algorithm described below from 
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
    
    First version; this doesn't properly account for different healing and transmittal rates, and is slow
    """
    t = 0
    print("beginning infection")
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

        u1 = np.random.random()
        
        if N_active > 0:
            
            prob_to_heal = N_active/(N_active+(lambda_avg*N_exposed))
    
            if u1 <= prob_to_heal:
                
                site_to_heal = np.random.randint(0, N_active)
                is_active[active_nodes[site_to_heal]] = 0
                #print(f"healed site {active_nodes[site_to_heal]}")
            else:
                infection_chances_list = np.array([sum([G.weight(u,v) for v in G.iterNeighbors(u)]) for u in exposed_edges[:,0]])
                infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                exposed_sites = exposed_edges[:, 1]
    
        
                site_to_infect = np.random.choice(exposed_sites, p=infection_chances_list)
                is_active[site_to_infect] = 1
                #print(f"infected site {site_to_infect}")
            
            active_nodes = np.array([u for u in G.iterNodes() if is_active[u]])
            #print(f"Active: {active_nodes} ---- t = {t}")
                
            t += 1/(N_active+lambda_avg*N_exposed)
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}")
            t = t_max
        
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%800 == 0:
            print(f"t={t} --- {N_active} nodes are active")
            visualize(G, active_nodes=active_nodes)
        
    print(f"{N_active} nodes are active at t={t}")

def DCP_v1(G, t_max = 1, random_start=True, lattice = True, torus = True):
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
    Same as v0, but doesn't recompute the active nodes list in every timestep
    """
    t = 0
    print("beginning infection")
    
    #all_nodes = np.array([u for u in G.iterNodes()])
    avg_list_time = np.array([])
    avg_rest_time = np.array([])
    active_nodes = np.array([u for u in G.iterNodes() if is_active[u] == 1])
    
    
    while t < t_max:
        t0=time.time_ns()
        
        exposed_edges = np.array([
                np.array([u, v]).T 
                for u in active_nodes #outer
                for v in G.iterNeighbors(u) #inner
            ])
        
        N_active = len(active_nodes)
        N_exposed = len(exposed_edges)# the number of exposed nodes is the same as the number of edges connected to active sites
        t1=time.time_ns()
        
        lambda_avg = (G.totalEdgeWeight()/G.numberOfEdges())

        u1 = np.random.random()
        
        if N_active > 0:
            
            #random_site = active_nodes[np.random.randint(0,N_active)]
            #rand_site_lambdas = [G.weight(random_site,v) for v in G.iterNeighbors(random_site)]
            #rand_site_lambda_avg = sum(rand_site_lambdas)/len(rand_site_lambdas)
            
            heal_chances = np.array([healing_factor[u] for u in active_nodes])
            heal_pmf = heal_chances*(1/sum(heal_chances))

            
            #prob_to_heal = (1/(1+(rand_site_lambda_avg)))*healing_factor[random_site]
            prob_to_heal = N_active/(N_active+(lambda_avg*N_exposed))
            
            if u1 <= prob_to_heal:
                
                site_to_heal = np.random.choice(active_nodes, p=heal_pmf)
                
                is_active[site_to_heal] = 0
                active_nodes = active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
            else:
                
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                #infection_chances_list = np.array([sum([G.weight(u,v) for v in G.iterNeighbors(u)]) for u in exposed_edges[:,0]]) #this double (or more) counts the sites sometimes
                edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                exposed_sites = exposed_edges[:, 1]
    
        
                site_to_infect = np.random.choice(exposed_sites, p=transmission_pmf)
                is_active[site_to_infect] = 1
                active_nodes = np.append(active_nodes, site_to_infect)
                #print(f"infected site {site_to_infect}")
            
            #active_nodes = np.array([u for u in G.iterNodes() if is_active[u]])
            #print(f"Active: {active_nodes} ---- t = {t}")
                
            t += 1/(N_active+lambda_avg*N_exposed)
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}")
            t = t_max
        
        """                         #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%800 == 0:
            print(f"t={t} --- {N_active} nodes are active")
            visualize(G, active_nodes=active_nodes)
        """
        print(f"t={t} --- {N_active} nodes are active")
        visualize(G, active_nodes=active_nodes, node_size=500)
        print(active_nodes)
        t2 = time.time_ns()
        
        avg_list_time=np.append(avg_list_time, t1-t0)
        #print(t1-t0)
        avg_rest_time=np.append(avg_rest_time, t2-t1)
        #print(t2-t1)
        #print(avg_list_time)
        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps")
    print(np.average(avg_list_time)/1000000000)
    print(np.average(avg_rest_time)/1000000000)

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
        - Don't recompute the active nodes 
        - It is ok to recompute edges; the computation combines sets (fast) and 
        networkit calls (fast) so it already takes on the order of 10^-5 
        seconds and isn't worth much imrpovement
    
    """
    t = 0
    print("beginning infection")
    
    active_nodes = set([u for u in G.iterNodes() if is_active[u] == 1])
    
    
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
        
        lambda_avg = np.average(exposed_edges)

        u1 = np.random.random()
        
        if N_active > 0:
            #this is the only time we need active_nodes to be ordered, so we make a list for it
            active_nodes_list = list(active_nodes)
            heal_chances = np.array([healing_factor[u] for u in active_nodes_list])
            #convert to pmf
            heal_pmf = heal_chances*(1/sum(heal_chances))

            prob_to_heal = N_active/(N_active+(lambda_avg*N_exposed))
            
            if u1 <= prob_to_heal:
                #for each node that could heal, choose according to pmf of mu values
                site_to_heal = np.random.choice(active_nodes_list, p=heal_pmf)
                
                is_active[site_to_heal] = 0
                active_nodes.remove(site_to_heal)
                #= active_nodes[active_nodes!=site_to_heal]
                #print(f"healed site {active_nodes[site_to_heal]}")
            else:
                
                #For each edge that /could/ transmit an infection, the probability of choosing is proportional to the edge's lambda value
                edge_transmission_chances = np.array([G.weight(u,v) for u,v in exposed_edges])
                transmission_pmf = edge_transmission_chances*(1/sum(edge_transmission_chances))
                #infection_chances_list = infection_chances_list*(1/sum(infection_chances_list))
                
                #print(f"active are {active_nodes}")
                exposed_sites = exposed_edges[:, 1]
    
        
                site_to_infect = np.random.choice(exposed_sites, p=transmission_pmf)
                is_active[site_to_infect] = 1
                active_nodes.add(site_to_infect)# = np.append(active_nodes, site_to_infect)

            
            t += 1/(N_active+lambda_avg*N_exposed)
            step_counter+=1
        else:
            print(f"Entered the absorbing state after {step_counter} timesteps at t = {t}")
            break
            #t = t_max
        
                                 #THIS BLOCK VISUALIZES THE DCP AT VARIOUS STEPS
        if step_counter%5000 == 0:
            print(f"t={t} --- {N_active} nodes are active")
            #visualize(G, active_nodes=active_nodes)
        

        
    print(f"{N_active} nodes are active at t={t} after {step_counter} steps")

    



"""
NEXT STEPS:
    Work on something that finds the phase transition lambda value given a graph G


"""
#x = sdrg(*generate_graph(500, 50, 20, 0.1, 0.01))

