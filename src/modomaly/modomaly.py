"""
Basic module for anomaly detection leveraging known community detection
methods.

This module provides functions for transforming time series data to
graphs/networks and running community detection algorithms on them to discover
community structures. Additional functions provide further community refinement
options.
"""

__all__ = [
        'graphify',
        'partition'
        # 'refine_parts'
        ]

import logging

import networkx as nx 
import pandas as pd 

modlog = logging.getLogger(__name__)

def graphify(df, xgap=0.4, ygap=40.0, epsilon=1.0):
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
    print(x)
    y = list(df['values'])
    n = len(x)

    edges = [(i,j) for i in range(n-1) for j in range(i+1, n)]

    modlog.info("Constructing graph G")
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n-1):
        for j in range(i+1,n):
            # weigh edge 1 if:
            # nodes apart no more than ygap 
            if abs(y[i]-y[j]) < ygap:
                # nodes apart no more than xgap
                if abs(x[j]-x[i]) < xgap: 
                    w = 1
                # or node j is close to at least one neighbor x of i
                elif len([v for v in G.neighbors(i) if x[j]-x[v] < epsilon*xgap]) > 0:
                    w = 1   # weigh edge 1
                else:
                    w = 0   # weigh edge 0 
            else:
                w = 0
            if w > 0:
                G.add_edge(i,j)
    
    modlog.info("Done constructing graph G")
    return G

def partition(G, alg="louvain"):
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
            communities = nx.community.louvain_communities(G)
            modularity = nx.community.modularity(G, communities)
            modlog.info("Done partitioning G")
        else:
            modlog.exception(NotImplementedError("Only 'louvain' community detection method implemented at this point"))
    
    return (communities, modularity)


