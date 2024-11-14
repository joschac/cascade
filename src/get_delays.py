from collections import deque
import numpy as np
import copy
import pandas as pd
import sys
from typing import List

from . import util
from . import model
from . import dataloader as dl
from . import our_gloabls

# Add the directory. Replace '/path/to/directory' with your directory path.
from .alarm import Alarm


class EventDelays(np.ndarray):
    def __new__(cls, event_delays, metadata=None):
        # Create the ndarray instance of event_delays. 
        # We need to ensure it's an array first.
        obj = np.asarray(event_delays).view(cls)
        
        # Add the additional attributes to the created instance
        obj.metadata = metadata

        # Return the newly created object
        return obj

    def __array_finalize__(self, obj):
        # This method is called whenever the ndarray is created.
        # We'll set default values for our custom attributes here.
        # But if obj is an instance of our class, we'll use its attributes.
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)
        
    
def getAdjList(Adjmatrix):
    _topology_list = [np.nonzero(Adjmatrix[:, d])[0] for d in range(len(Adjmatrix))] #list of neighbors for each device
    return _topology_list

def get_event_delays_old(data, topology_matrix, N=10, M=10, delta_t=1000, save_path=None, long_format=False, min_delay=1, effect_event_types=None):
    """
    Computes the event delays for each pair of events in the data.
    Parameters
    __________________
    data: numpy array: 
    topology_matrix: numpy array of shape (nodes, nodes), topology matrix
    N: number of event types
    M: number of devices
    delta_t: time window for computing event delays
    save_path: path to save the result
    min_delay: consider only cause events with delay >= min_delay (if 0 allows for instantaneous effects)
    effect_event_types: list of effect event types to calculate delays for. all other edges event delays are left empty.
    Returns
    _________________
    result: pandas dataframe with lists of event delays of potential causes for each effect event. if long_format=False [rows: cause, columns: effect, and values: event delays.]. if long_format=True [each row has columns: cause, effect, and event delays. ]
    """
    eventtypes =  list(range(N))
    devicetypes = list(range(M))
    if not effect_event_types:
        effect_event_types = eventtypes

    

    cause_start_times = [[[] for e in eventtypes] for d in devicetypes] #3D array of shape (nodes, events, [start timestamps])
    effect_start_times = [[[] for e in eventtypes] for d in devicetypes]
    _topology_list = getAdjList(topology_matrix)

    #for i in range(len(data)):
    #    d,_,start_time, a, explained = data[i, :5]
    #    cause_start_times[d][a].append(start_time)
    #    if explained == -1:
    #        effect_start_times[d][a].append(start_time)

    
    unexplained_events = data[data[:, 4]==-1]

    for d in devicetypes:
        for a in eventtypes:
            cause_start_times[d][a] = data[np.logical_and(data[:,0]==d, data[:,3]==a), 2]
        for a in effect_event_types:
            effect_start_times[d][a] = unexplained_events[np.logical_and(unexplained_events[:,0]==d, unexplained_events[:,3]==a), 2] #unexplained events

    result = [[[] for e in eventtypes] for e in eventtypes]
    for A in eventtypes:
        for B in effect_event_types:
            for node in devicetypes:
                if len(effect_start_times[node][B]) == 0:
                        continue
                neighborhood = list(_topology_list[node]) + [node]
                for neighbor in neighborhood:
                    if len(cause_start_times[neighbor][A]) == 0:
                        continue
                    new_min_delay = 1 if min_delay == 0 and A == B and neighbor == node else min_delay
                    tBs = np.array(effect_start_times[node][B])
                    lefts = np.searchsorted(cause_start_times[neighbor][A], tBs-delta_t, side='left')
                    rights = np.searchsorted(cause_start_times[neighbor][A], tBs-new_min_delay+1, side='left') 
                    to_iterate = np.nonzero(np.logical_and(lefts < len(cause_start_times[neighbor][A]), rights > lefts))
                    if len(to_iterate[0]) > 0:
                        for i in to_iterate[0]:
                            tB = tBs[i]
                            left, right = lefts[i], rights[i]
                            delays = tB - np.array(cause_start_times[neighbor][A][left:right])
                            delays = delays[::-1] #reverse delays to be in ascending order
                            delays = EventDelays(delays, metadata=Alarm(node,B,tB)) #add metadata to delays
                            result[A][B].append(delays)

    result = pd.DataFrame(result, columns=eventtypes, index=eventtypes)
    if long_format:
        #convert to format: cause, effect, event delay
        result = pd.DataFrame(result.stack()).reset_index()
        result.columns = ['cause', 'effect', 'events delays']
        # set column type to int
        result['cause'] = result['cause'].astype(int)
        result['effect'] = result['effect'].astype(int)
    if save_path is not None:
        result.to_pickle(save_path)

    return result


def get_event_delays(data, topology_matrix, N=10, M=10, delta_t=1000, save_path=None, long_format=False, min_delay=0 if our_gloabls.instant_effects else 1 , effect_event_types=None, check_sorted=True):
    """
    Computes the event delays for each pair of events in the data.
    Parameters
    __________________
    data: numpy array: 
    topology_matrix: numpy array of shape (nodes, nodes), topology matrix
    N: number of event types
    M: number of devices
    delta_t: time window for computing event delays
    save_path: path to save the result
    min_delay: consider only cause events with delay >= min_delay (if 0 allows for instantaneous effects)
    effect_event_types: list of effect event types to calculate delays for. all other edges event delays are left empty.
    Returns
    _________________
    result: pandas dataframe with lists of event delays of potential causes for each effect event. if long_format=False [rows: cause, columns: effect, and values: event delays.]. if long_format=True [each row has columns: cause, effect, and event delays. ]
    """
    min_delay=0 if our_gloabls.instant_effects else 1
    eventtypes =  list(range(N))
    devicetypes = list(range(M))
    if not effect_event_types:
        effect_event_types = eventtypes


    cause_start_times = [[[] for e in eventtypes] for d in devicetypes] #3D array of shape (nodes, events, [start timestamps])
    effect_start_times = [[[] for e in eventtypes] for d in devicetypes]
    _topology_list = getAdjList(topology_matrix)

    #Check if sorted
    if check_sorted:
        if not (all(data[i,2] <= data[i + 1, 2] for i in range(len(data[:,2])-1))):
            raise RuntimeError("Input to get_event_delays not sorted.")

    unexplained_events = data[data[:, 4]==-1]

    for d in devicetypes:
        for a in eventtypes:
            cause_start_times[d][a] = data[np.logical_and(data[:,0]==d, data[:,3]==a), 2]
        for a in effect_event_types:
            effect_start_times[d][a] = unexplained_events[np.logical_and(unexplained_events[:,0]==d, unexplained_events[:,3]==a), 2] #unexplained events

    result = [[[] for e in eventtypes] for e in eventtypes]
    for A in eventtypes:
        for B in effect_event_types:
            for node in devicetypes:
                TAs = cause_start_times[node][A]
                if len(TAs) == 0:
                        continue
                neighborhood = list(_topology_list[node]) + [node]
                for neighbor in neighborhood:
                    TBs = effect_start_times[neighbor][B]
                    if len(TBs) == 0:
                        continue
                    
                    new_min_delay = 1 if min_delay == 0 and A == B and neighbor == node else min_delay
                    
                    tAs = np.array(TAs)
                    lefts = np.searchsorted(TBs, tAs+new_min_delay, side='left')
                    rights = np.searchsorted(TBs, tAs+delta_t, side='left') 
                    to_iterate = np.nonzero(np.logical_and(lefts < len(TBs), rights > lefts))
                    if len(to_iterate[0]) > 0:
                        for i in to_iterate[0]:
                            tA = TAs[i]
                            left, right = lefts[i], rights[i]
                            delays = np.array(TBs[left:right]) - tA
                            #delays = delays[::-1] #reverse delays to be in ascending order
                            delays = EventDelays(delays, metadata=Alarm(node,A,tA)) #add metadata to delays
                            result[A][B].append(delays)

    result = pd.DataFrame(result, columns=eventtypes, index=eventtypes)
    if long_format:
        #convert to format: cause, effect, event delay
        result = pd.DataFrame(result.stack()).reset_index()
        result.columns = ['cause', 'effect', 'events delays']
        # set column type to int
        result['cause'] = result['cause'].astype(int)
        result['effect'] = result['effect'].astype(int)
    if save_path is not None:
        result.to_pickle(save_path)

    return result


def test_get_delays():
    save_path = 'event_delays.pkl'
    topology_matrix = np.load(r'./data/competition/HuaweiVirus/sample/topology.npy')
    alarms_df = pd.read_csv(r'./data/competition/HuaweiVirus/sample/alarm.csv')
    causal_prior= np.load(r'./data/competition/HuaweiVirus/sample/causal_prior.npy')
    base_cost = util.get_base_alarm_cost(alarms_df)
    alarms = alarms_df['alarm_id'].unique()

    G, device_dict = dl.all_data_graph(alarms_df, topology_matrix)
    m = model.Model(G,len(alarms), len(alarms_df), device_dict)
    delays = get_event_delays(m.all_alarms, topology_matrix, len(causal_prior), len(topology_matrix), delta_t=our_gloabls.max_delay, long_format=True, save_path=None)
    print(delays)

if __name__ == '__main__':
    test_get_delays()
