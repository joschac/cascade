import pandas as pd
import numpy as np
import networkx as nx
import pickle
import logging
# logging.basicConfig(level=logging.DEBUG)
import argparse

from . import our_gloabls
from . import get_delays
from . import util
from . import score_all_edges
from . import model
from . import dataloader as dl
from . import find_topological_order

def introduce_cycle(edges, edge_cand):
    bin_rep = np.vectorize(lambda x: 0 if x==None else 1)(edges)
    DiG = nx.DiGraph(bin_rep)
    DiG.add_edge(edge_cand.cause, edge_cand.effect)
    try:
        cycles = nx.find_cycle(DiG, edge_cand.cause, orientation='original')
        return True
    except nx.NetworkXNoCycle: 
        return False
    

def set_causal_prior_edges(causal_prior, delays, candidates, tested_candidates):
    causal_edges = np.where(causal_prior == 1)
    for cause, effect in zip(causal_edges[0], causal_edges[1]):
        # if cause != effect: # why was this here?
        i = delays.where((delays['effect'] == effect) & (delays['cause'] == cause)).dropna().index[0] 
        if not tested_candidates[i]:
            candidates[i].gain_est = -np.inf # hack to add prior edges first


def search(alarms_df:pd.DataFrame, topology_matrix, causal_prior, no_topo=False):

    raise NotImplementedError("Greedy search deprecated.")

    base_cost = util.get_base_alarm_cost(alarms_df)
    alarms = np.arange(causal_prior.shape[0])

    G, device_dict = dl.all_data_graph(alarms_df, topology_matrix)
    m = model.Model(G,len(alarms),base_cost, len(alarms_df), device_dict)

     #assume sorted by start times, to avoid sorting each time in get_delays
    delays = get_delays.get_event_delays(m.all_alarms, topology_matrix, len(causal_prior), len(topology_matrix), delta_t=our_gloabls.max_delay, long_format=True, save_path=None)
    candidates = score_all_edges.score_all_edges(delays,base_cost,m.alarm_counts,m)
    topological_order = find_topological_order.find_topological_order_from_candidates(candidates)
    tested_candidates = np.zeros(len(candidates), dtype=bool)

    set_causal_prior_edges(causal_prior, delays, candidates, tested_candidates)
    
    non_causal_edges = np.where(causal_prior == 0)
    for cause, effect in zip(non_causal_edges[0], non_causal_edges[1]):
        if cause != effect:
            i = delays.where((delays['effect'] == effect) & (delays['cause'] == cause)).dropna().index[0]
            tested_candidates[i] = True
    
    
    # add edges from causal prior first?
    # skip edges that would violate causal prior
    while True:
        while True:
            edgeCand_pt = candidates.argmin() # using simple min instead of heap, since we have to re - heapify after each edge addition anyway

            if m.edges[candidates[edgeCand_pt].cause,candidates[edgeCand_pt].effect] == None and\
               tested_candidates[edgeCand_pt] == False and\
                not introduce_cycle(m.edges, candidates[edgeCand_pt])\
                and (topological_order.index(candidates[edgeCand_pt].effect) > topological_order.index(candidates[edgeCand_pt].cause) or no_topo):
                break
            elif candidates[edgeCand_pt].gain_est > 0:
                break
            else:
                candidates[edgeCand_pt].gain_est = 1 
        edgeCand = candidates[edgeCand_pt]
        logging.debug(f"testing edge number {tested_candidates.sum()} / {tested_candidates.size}")
        if edgeCand.gain_est < 0:
            edge = edgeCand.get_edge()
            alarms = m.test_add_edge(edge)
            # delays = get_delays.update_event_delays(delays, alarms)
            # delays = get_delays.get_event_delays(m.all_alarms, topology_matrix, len(causal_prior), len(topology_matrix), delta_t=our_gloabls.max_delay, long_format=True, save_path=None)
            new_delays = get_delays.get_event_delays(m.all_alarms, topology_matrix, len(causal_prior), len(topology_matrix), delta_t=our_gloabls.max_delay, long_format=True, save_path=None, effect_event_types=[edge.effect])
            new_delays = new_delays.where((new_delays['effect'] == edge.effect)).dropna()
            delays.loc[new_delays.index] = new_delays
            # update_index = delays.where((delays['effect'] == edge.effect)).dropna().index
            update_index = new_delays.index
            candidates[update_index] = score_all_edges.score_all_edges(delays.loc[update_index],base_cost, m.alarm_counts,m)
            candidates[edgeCand_pt].gain_est = 1 # mark as already tested 
            tested_candidates[edgeCand_pt] = True
            set_causal_prior_edges(causal_prior, delays, candidates, tested_candidates) 
        else:
            break

    return m


def topological_search(alarms_df:pd.DataFrame, topology_matrix, causal_prior, init_with_self_loops=False):
    unique_alarms = np.arange(causal_prior.shape[0])

    G, device_dict = dl.all_data_graph(alarms_df, topology_matrix)
    if init_with_self_loops:
        m = set_matrix_as_model(alarms_df, topology_matrix, np.identity(len(unique_alarms)))
    else:
        m = model.Model(G,len(unique_alarms), len(alarms_df),  device_dict)

    delays = get_delays.get_event_delays(m.all_alarms, topology_matrix, len(causal_prior), len(topology_matrix), delta_t=our_gloabls.max_delay, long_format=True, save_path=None)
    candidates = score_all_edges.score_all_edges(delays,m.base_cost,m.alarm_counts,m)
    candidate_cause  = np.vectorize(lambda candidate: candidate.cause)(candidates)
    candidate_effect = np.vectorize(lambda candidate: candidate.effect)(candidates)

    tested_candidates = np.zeros(len(candidates), dtype=bool) # not needed anymore?
    
    added_nodes = set()
    while len(added_nodes) < len(unique_alarms): 
        next_node = find_topological_order.get_next_node(candidates, added_nodes)
        m.test_remove_effect_edges(next_node)
        added_nodes.add(next_node)
        next_node_candidates_pt = np.where((candidate_cause == next_node) & (np.vectorize(lambda x: x not in added_nodes)(candidate_effect)))
        next_node_candidates = candidates[next_node_candidates_pt]
        its_time_to_stop = False
        while len(next_node_candidates) > 0 and not its_time_to_stop:
            
            edgeCand_pt = next_node_candidates.argmin()
            edgeCand = next_node_candidates[edgeCand_pt]
            edge = edgeCand.get_edge()
            
            removed_edges = m.remove_effect_edges(edgeCand.effect) if edgeCand.gain_est < np.inf else [] 

            if len(removed_edges) > 0:
                new_delays = get_delays.get_event_delays(m.all_alarms, topology_matrix, len(causal_prior), len(topology_matrix), delta_t=our_gloabls.max_delay, long_format=True, save_path=None, effect_event_types=[edge.effect])
                new_delays = new_delays.where((new_delays['effect'] == edge.effect)).dropna()
                delays.loc[new_delays.index] = new_delays
                # update_index = delays.where((delays['effect'] == edge.effect)).dropna().index
                update_index = new_delays.index
                candidates[update_index] = score_all_edges.score_all_edges(delays.loc[update_index],m.base_cost, m.alarm_counts,m)
                next_node_candidates = candidates[next_node_candidates_pt]
                iteration_candidates_pt = np.where((np.isin(candidate_cause,removed_edges)) & (candidate_effect == edgeCand.effect))
                iteration_candidates = candidates[iteration_candidates_pt]
                seen_causes = set()
                while len(iteration_candidates) > 0:
                    
                    while True:
                        iteration_edgeCand_pt = iteration_candidates.argmin()
                        if iteration_candidates[iteration_edgeCand_pt].gain_est < next_node_candidates[edgeCand_pt].gain_est:
                            iteration_edgeCand = iteration_candidates[iteration_edgeCand_pt]
                        else:
                            iteration_edgeCand = next_node_candidates[edgeCand_pt]
                        if iteration_edgeCand.gain_est == np.inf:
                            break
                        elif iteration_edgeCand.cause not in seen_causes:
                            seen_causes.add(iteration_edgeCand.cause)
                            break
                        else:
                            iteration_edgeCand.gain_est = np.inf

                    if iteration_edgeCand.gain_est < 0:
                        iteration_edge = iteration_edgeCand.get_edge()
                        m.test_add_edge(iteration_edge)

                        new_delays = get_delays.get_event_delays(m.all_alarms, topology_matrix, len(causal_prior), len(topology_matrix), delta_t=our_gloabls.max_delay, long_format=True, save_path=None, effect_event_types=[edge.effect])
                        new_delays = new_delays.where((new_delays['effect'] == edge.effect)).dropna()
                        delays.loc[new_delays.index] = new_delays
                        # update_index = delays.where((delays['effect'] == edge.effect)).dropna().index
                        update_index = new_delays.index
                        candidates[update_index] = score_all_edges.score_all_edges(delays.loc[update_index],m.base_cost, m.alarm_counts,m)
                        next_node_candidates = candidates[next_node_candidates_pt]
                        iteration_candidates = candidates[iteration_candidates_pt]
                    else:
                        break

                    iteration_edgeCand.gain_est = np.inf # mark as already tested
                next_node_candidates[edgeCand_pt].gain_est = np.inf  # mark as already tested
            elif edgeCand.gain_est < 0:
                alarms = m.test_add_edge(edge)
                
                new_delays = get_delays.get_event_delays(m.all_alarms, topology_matrix, len(causal_prior), len(topology_matrix), delta_t=our_gloabls.max_delay, long_format=True, save_path=None, effect_event_types=[edge.effect])
                new_delays = new_delays.where((new_delays['effect'] == edge.effect)).dropna()
                delays.loc[new_delays.index] = new_delays
                # update_index = delays.where((delays['effect'] == edge.effect)).dropna().index
                update_index = new_delays.index
                candidates[update_index] = score_all_edges.score_all_edges(delays.loc[update_index],m.base_cost, m.alarm_counts,m)
                next_node_candidates = candidates[next_node_candidates_pt]
                next_node_candidates[edgeCand_pt].gain_est = np.inf 

            else:
                break
    return m

def set_matrix_as_model(alarms_df:pd.DataFrame, topology_matrix, causal_matrix, skip_reassign=False):
    unique_alarms = np.arange(causal_matrix.shape[0])

    G, device_dict = dl.all_data_graph(alarms_df, topology_matrix)
    m = model.Model(G,len(unique_alarms), len(alarms_df),  device_dict)
    
    delays = get_delays.get_event_delays(m.all_alarms, topology_matrix, len(causal_matrix), len(topology_matrix), delta_t=our_gloabls.max_delay, long_format=True, save_path=None)
    candidates = score_all_edges.score_all_edges(delays,m.base_cost,m.alarm_counts,m)
    tested_candidates = np.zeros(len(candidates), dtype=bool)

    set_causal_prior_edges(causal_matrix, delays, candidates, tested_candidates)
    while True:
        edgeCand_pt = candidates.argmin()
        edgeCand = candidates[edgeCand_pt]
        if edgeCand.gain_est == -np.inf:
            edge = edgeCand.get_edge()
            m.edges[edge.cause, edge.effect] = edge
            edgeCand.gain_est = 1 # mark as already added
        else:
            break
    if not skip_reassign: #fix true explanation here ? 
        m.reassign_all_alrams_and_refit()
        print("Length under set causal prior: ", m.compute_length())
    return m


