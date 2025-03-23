import logging
import os
import modomaly 
import datetime 

import pandas as pd 


TEST_DATA_LOCATION=os.path.join("data")
TEST_FILENAME="lidar_5.csv"
# Configure the root logger
logging.basicConfig(
        filename="main.log",
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # handlers=[logging.StreamHandler()]
)

# Get the root logger
logger = logging.getLogger(__name__)
logger.info("'Main' started")

df = pd.read_csv(os.path.join(TEST_DATA_LOCATION, TEST_FILENAME))
df['values'] = df['distance'].fillna(0).astype(int)
df['datetimestr'] = df[['date', 'time']].apply(lambda x: ' '.join(x), axis=1)
df['datetime'] = pd.to_datetime(df['datetimestr'], format='%Y-%m-%d %H:%M:%S.%f')
df['unix_time'] = df['datetime'].apply(lambda x: datetime.datetime.timestamp(x))
G = modomaly.graphify(df)
print(G, flush=True)
P, m = modomaly.partition(G)
print(P, flush=True)
for p in P:
    for i in p:
        print(df.iloc[i]['values'], end=' ')
    print()
    print()


logger.info("'Main' finished")
