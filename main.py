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

np.random.seed(42)

#G = nx.partial_duplication_graph(50, 4, 0.4, 0.6)

def generate_graph(size, mean_cluster_size, cluster_size_std_dev, p_in, p_out):
    G = nx.gaussian_random_partition_graph(size, mean_cluster_size, cluster_size_std_dev, p_in, p_out)
    
    node_atts = {x: (False, np.random.random()) for x in range(len(G.nodes))}
    #print(list(G.edges))
    edge_atts = {x: round(np.random.random(),4) for x in list(G.edges)}
    
    #print(edge_atts)
    
    nx.set_node_attributes(G, node_atts, "statemu")
    
    # for edge in list(G.edges):
    #     G.edges[edge]["weight"] = 0.1#round(np.random.random(),3)
    
    nx.set_edge_attributes(G, edge_atts, "lambda")


    #print(nx.get_node_attributes(G, "statemu"))
    #nx.draw(G)
    
    
    # 1. Compute node positioning layout
    pos = nx.spring_layout(G)
    
    # 2. Draw the base nodes and lines
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_labels(G, pos)
    
    # 3. Dynamic styling: set line thickness proportional to its weight attribute
    edge_widths = [data["lambda"] for u, v, data in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=edge_widths)
    
    # 4. Overlay numerical text labels onto the edges
    edge_labels = nx.get_edge_attributes(G, "lambda")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
    
    #plt.show()
    
    return [G, node_atts, edge_atts]
    
    
def sdrg(G, node_atts, edge_atts):
    max_mu = 0
    for i in range(node_atts)
        if node_atts
    
    
    
    
x = sdrg(*generate_graph(500, 50, 20, 0.1, 0.01))

