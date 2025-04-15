import logging
import os
import modomaly 
import datetime 

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd 

from statistics import mean

TEST_DATA_LOCATION=os.path.join("data")

# Get the root logger
logger = logging.getLogger(__name__)
# Configure the root logger
logging.basicConfig(
        filename="main.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # handlers=[logging.StreamHandler()]
)
logger.info("'Main' started")

for k in range(1):
    
    TEST_FILENAME="lidar_"+str(k)+".csv" # each file is one excerpt containing an overtake
    COLOR=['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    logger.info("{}: Processing event".format(k))

    # --------------------------------- PARTITIONING ---------------------------------
    df = pd.read_csv(os.path.join(TEST_DATA_LOCATION, TEST_FILENAME))
    df['values'] = df['distance'].fillna(0).astype(int)
    df['datetimestr'] = df[['date', 'time']].apply(lambda x: ' '.join(x), axis=1)
    df['datetime'] = pd.to_datetime(df['datetimestr'], format='%Y-%m-%d %H:%M:%S.%f')
    df['unix_time'] = df['datetime'].apply(lambda x: datetime.datetime.timestamp(x))

    # Call modularity based module functions
    G = modomaly.make_graph(df)
    P, m = modomaly.get_partition(G)
    
    # Try spectral method
    # so = nx.spectral_ordering(G)
    # fv = nx.fiedler_vector(G) # eigenvector corresponding to the 2nd smallest evalue (for L = D - A, that's the first non-zero evalue)
    cc = list(nx.connected_components(G))
    # eL = nx.laplacian_spectrum(G)
    print(len(cc))
    
    # ---------------------------------- OUTPUT ------------------------------------
    logger.info("{}: Plotting figures".format(k))
    partition_info = []

    for p in P: # part in Partition
        p = list(p)
        partition_info.append({'times': list(df.iloc[p]['unix_time']), 'values': list(df.iloc[p]['values'])})
    
    # plot each partition in different color
    for i, p in enumerate(partition_info):
        c = COLOR[i%len(COLOR)]
        plt.scatter(p['times'], p['values'], color=c)
    plt.savefig(os.path.join("out", "lidar_"+str(k)+".png"))
    plt.clf()
    
    # plot biggest low partition in red, rest in blue
    low_parts = []
    for i, p in enumerate(partition_info):
        if mean(p['values']) < 500:
           low_parts.append(i)

    lowest_substantial_part = 0
    max_len = 0
    max_i = 0
    for i in low_parts:
        n = len(partition_info[i]['values'])
        if n > max_len:
            max_len = n
            max_i = i

    for i, p in enumerate(partition_info):
        c = 'tab:blue'
        if i == max_i:
            c = 'tab:orange'
        plt.scatter(p['times'], p['values'], color=c)
    plt.savefig(os.path.join("out", "weighted-edges", "lidar_bi-color-"+str(k)+".png"))
    plt.clf()

    for i, p in enumerate(partition_info):
        c = COLOR[i%len(COLOR)]
        plt.scatter(p['times'], p['values'], color=c)
    plt.savefig(os.path.join("out", "weighted-edges", "lidar_"+str(k)+".png"))
    plt.clf()

logger.info("'Main' finished")
