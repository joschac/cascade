import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from . import device
from . import alarm

def transform_from_float(alarms:pd.DataFrame):
    multiplier = 1/np.sort(np.diff(np.sort(alarms['start_timestamp'])))[0]
    alarms['start_timestamp'] = (alarms['start_timestamp']*multiplier).astype(int)
    return alarms, multiplier

def get_full_connected_topology_matrix(alarms:pd.DataFrame):
    N = alarms['device_id'].nunique()
    topology_matrix = np.ones((N,N), dtype=int)
    np.fill_diagonal(topology_matrix, 0) # self loops are dealt with separately
    return topology_matrix

def get_empty_topology_matrix(alarms:pd.DataFrame):
    N = alarms['device_id'].nunique()
    topology_matrix = np.zeros((N,N), dtype=int)
    return topology_matrix

def get_empty_causal_prior(alarms:pd.DataFrame, true_graph_path:str):
    N = alarms['alarm_id'].nunique() if true_graph_path is None else np.load(true_graph_path).shape[0]
    causal_prior = np.zeros((N,N), dtype=int)
    causal_prior.fill(-1)
    return causal_prior

# return data as dictionary of dictionaries
# first key: device_id
# second key: alarm_id
# for each device there is information when an alarm was triggered
def read_alarm_data(timeseries_path, topology_path, prior_path):
    df = pd.read_csv(timeseries_path,sep=",")
    #get unique values
    alarm_ids = df["alarm_id"].unique()
    device_ids = df["device_id"].unique()

    # for each alarm create list of starting time
    data_structure = {}
    for id in alarm_ids:
        data_structure[id] = []
    
    # for each device_id array of alarm start timestamps
    alarm_start_data = {}
    for id in device_ids:
        # for each alarm create list of starting time
        alarm_start_data[id] = data_structure.copy()
    
    # add alarm start time to the list
    for index, row in df.iterrows():
        alarm_start_data[row["device_id"]][row["alarm_id"]].append(row["start_timestamp"])

    device_connectivity = np.load(topology_path) 
    device_connectivity_graph = nx.DiGraph(device_connectivity, create_using=nx.DiGraph)
    prior = np.load(prior_path)
    # prior possible values 1, 0, -1 indicating causal, not causal, unknown
    # for now ignore not causal
    candidate_parent_nodes = {}
    for id in alarm_ids:
        candidates = []
        for parent in alarm_ids:
            if id != parent:
                # there does not exist an edge between id and parent
                if prior[parent,id] == 0:
                    continue
                # there exists an edge between id and parent do nothing for now
                if prior[parent,id] == 1:
                    continue
                candidates.append(parent)
        candidate_parent_nodes[id] = candidates

    prior[prior == -1] = 0
    prior = nx.DiGraph(prior, create_using=nx.DiGraph)
    causal_graph_from_prior = nx.DiGraph()
    for edge in prior.edges():
        if prior[edge[0]][edge[1]]['weight'] == 1:
            causal_graph_from_prior.add_edge(edge[0], edge[1])
    return alarm_start_data, device_connectivity_graph, causal_graph_from_prior, candidate_parent_nodes

def get_alarms_df(timeseries_path):
    df = pd.read_csv(timeseries_path,sep=",")
    return df


def all_data_graph(alarms_df, topology_matrix):
    device_dict = {}
    for d_id in alarms_df['device_id'].unique():
        alarms_on_device = alarms_df[alarms_df['device_id']==d_id].sort_values(by='start_timestamp')[['alarm_id','start_timestamp']]
        alarms_on_device = alarms_on_device.apply(lambda x: alarm.Alarm(d_id,x['alarm_id'],x['start_timestamp']),axis=1).values
        device_dict[d_id] = device.Device(d_id,alarms_on_device)
    G = nx.Graph(topology_matrix)
    # replace notes with Object of type Device
    G = nx.relabel_nodes(G, lambda x: device_dict[x], copy=False)
    return G, device_dict