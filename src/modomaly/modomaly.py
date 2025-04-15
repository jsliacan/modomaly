"""
Basic module for anomaly detection leveraging known community detection
methods.

This module provides functions for transforming time series data to
graphs/networks and running community detection algorithms on them to discover
community structures. Additional functions provide further community refinement
options.
"""

__all__ = [
        'make_graph',
        'get_partition'
        # 'refine_parts'
        ]

import logging
import os 

import networkx as nx 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

modlog = logging.getLogger(__name__)

def make_graph(df, xgap=0.4, ygap=40.0, epsilon=1.0):
    """
    Return a networx Graph instance based on the time series in the dataframe.

    Parameters
    ----------
    df : dataframe
        DataFrame is expected to have two columns: `time` and `values`
    xgap : float
        Maximum time gap in the time series, in seconds. Key to deciding to
        place an edge or a non-edge when constructing the graph. Default is 0.4.
    ygap : float
        Maximum value gap in the time series. Key to deciding to place an edge
        or a non-edge when constructing the graph. Default is 40.0.
    epsilon : float
        Small quantity specifing a fraction of xgap used to determine
        second-order closeness between two vertices. Default is 1.0.
    
    Returns
    -------
    G : networkx.Graph
    """

    # a reasonable xgap for OTs is 20 lines at 50Hz, so pass xgap=0.4. 
    x = list(df['unix_time'])
    y = list(df['values'])
    print([i for i in range(len(y)) if y[i] < 400])
    n = len(x)

    edges = [(i,j) for i in range(n-1) for j in range(i+1, n)]

    modlog.info("Constructing graph G")
    G = nx.Graph()
    pos = {}
    for i in range(n):
        pos[i] = (i, y[i])
    G.add_nodes_from(pos.keys())
    
    x0 = x[0]

    for i in range(n-1):
        for j in range(i+1,n):
            # weigh edge 1 if:
            # nodes apart no more than ygap 
            
            w = (100/np.linalg.norm(np.array([x[i], y[i]]) - np.array([x[j], y[j]]), 2)**1.5)
            # print(w, flush=True)

            if abs(y[i]-y[j]) < ygap:
                # nodes apart no more than xgap
                if abs(x[j]-x[i]) < xgap: 
                    # w = 1
                    pass
                # or node j is close to at least one neighbor x of i
                elif len([v for v in G.neighbors(i) if x[j]-x[v] < epsilon*xgap]) > 0:
                    # w = 1   # weigh edge 1
                    pass
                else:
                    # w = 0   # weigh edge 0 
                    pass
            else:
                # w = 0
                pass
            if w > 0:
                G.add_edge(i,j, weight=w, resolution=0.1)
   
    # for e in G.edges(data=True):
    #     if 57 in e:
    #         print(e)

    # nx.draw(G, pos)
    # labels = nx.get_edge_attributes(G,'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # plt.savefig(os.path.join("out", "weighted-edges", "G-edges-1.png"))
    
    modlog.info("Done constructing graph G")
    return G

def get_partition(G, alg="louvain"):
    """
    Return a 2-tuple containing the list of communities and the modularity
    score.

    Parameters
    ----------
    G : networkx.Graph 
        Graph instance to be partitioned into communities.
    alg : optional str 
        Name of the algorithm to use. Currently limited to 'louvain'.
    
    Returns
    -------
    communities : list of lists of (int or float)
        List of subsets of vertices that correspond to communities in the graph.
    modularith : float
        Modularity score of the graph partition.
    """
    
    # TODO: implement more algorithm options, e.g. infomap, girvan-newman,
    # label-propagation, etc.

    # TODO: use 'resolution' parameter of the louvain algorithm

    communities = []
    modularity = 0
    if len(G.edges) == 0:
        modlog.error("Graph G has 0 edges. Cannot construct a valid partition.")
        communities = []
        modularity = 0
    else:
        if alg == "louvain":
            modlog.info("Partitioning G")
            communities = nx.community.louvain_communities(G, weight='weight')
            modularity = nx.community.modularity(G, communities)
            modlog.info("Done partitioning G")
        else:
            modlog.exception(NotImplementedError("Only 'louvain' community detection method implemented at this point"))
    
    return (communities, modularity)


