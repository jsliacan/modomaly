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
        'partition',
        'refine_parts'
        ]

import networkx as nx 
import pandas as pd 

def graphify(df, xgap=0.4, epsilon=1.0, ygap=40.0):
    """
    Return a networx Graph instance based on the time series in the dataframe.

    Parameters
    ----------
    df      :   dataframe
        DataFrame is expected to have two columns: `time` and `values`
    xgap    :   float
        Maximum time gap in the time series, in seconds. Key to deciding to
        place an edge or a non-edge when constructing the graph. Default is 0.4.
    epsilon :   float
        Small quantity specifing a fraction of xgap used to determine
        second-order closeness between two vertices. Default is 1.0.
    ygap    :   float
        Maximum value gap in the time series. Key to deciding to place an edge
        or a non-edge when constructing the graph. Default is 40.0.
    
    Returns
    -------
    G       :   networkx.Graph
    """

    # a reasonable xgap for OTs is 20 lines at 50Hz, so pass xgap=0.4. 
    x = list(df['time'])
    y = list(df['values'])
    n = len(x)

    edges = [(i,j) for i in range(n-1) for j in range(i+1, n)]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n-1):
        for j in range(i+1,n):
            # weigh edge 1 if:
            # nodes apart no more than ygap 
            if abs(y[i]-y[j]) < ygap:
                # nodes apart no more than xgap
                if x[j]-x[i] < xgap: 
                    w = 1
                # or node j is close to at least one neighbor x of i
                elif len([v for v in G.neighbors(i) if x[j]-x[v] < epsilon*xgap]) > 0:
                    w = 1   # weigh edge 1
                else:
                    w = 0   # weigh edge 0 
            else:
                w = 0

            G.add_edge(i,j, weight=w)

    return G

def partition(G, alg="louvain"):
    """
    Return a 2-tuple containing the list of communities and the modularity
    score.

    Parameters
    ----------
    G       :   networkx.Graph 
        Graph instance to be partitioned into communities.
    alg     :   optional str 
        Name of the algorithm to use. Currently limited to 'louvain'.
    
    Returns
    -------
    communities     : list of lists of (int or float)
        List of subsets of vertices that correspond to communities in the graph.
    modularith      : float
        Modularity score of the graph partition.
    """
    
    # TODO: implement more algorithm options, e.g. infomap, girvan-newman,
    # label-propagation, etc.

    # TODO: use 'resolution' parameter of the louvain algorithm

    communities = []
    modularity = 0

    
    if len(edges) == 0:
        print("Error: graph G has 0 edges. Cannot construct a valid partition.")
        communities = []
        modularity = 0
    else:
        communities = nx.community.louvain_communities(G)
        modularity = nx.community.modularity(G, communities)

    return (communities, modularity)

def test_modomaly(name):
    """
    Print 'Hello <name>' string with <name> from the argument.
    """

    print("Hello {}!".format(name))

