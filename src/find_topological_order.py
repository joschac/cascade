import numpy as np


def __find_topological_order(score_lookup, candidate_sinks, break_after_first=False): 
    topological_order = []
    while len(candidate_sinks) > 0:
        worst_best_gain = - np.inf
        worst_node = candidate_sinks[0]
        for node in candidate_sinks:
            node_best_gain = np.inf
            for potential_parent in candidate_sinks:
                if potential_parent == node:
                    continue
                score = score_lookup[(potential_parent,node)]
                if score < node_best_gain:
                    node_best_gain = score
            if node_best_gain > worst_best_gain:
                worst_best_gain = node_best_gain
                worst_node = node
        topological_order.append(worst_node)
        if break_after_first:
            break
        candidate_sinks.remove(worst_node)
    return topological_order


def find_topological_order_from_candidates(candidates):
    score_lookup = {}
    candidate_sinks = set()
    for cand in candidates:
        score_lookup[(cand.cause, cand.effect)] = cand.gain_est
        candidate_sinks.add(cand.cause)
    return __find_topological_order(score_lookup, list(candidate_sinks))

def get_next_node(candidates, added_notes=set()):
    candidate_sinks = set()
    unique_nodes = []
    for cand in candidates:
        if cand.cause not in unique_nodes and cand.cause not in added_notes:
            unique_nodes.append(cand.cause)
    score_lookup = np.zeros((len(unique_nodes), len(unique_nodes)))

    for cand in candidates:
        if cand.cause not in unique_nodes or cand.effect not in unique_nodes:
            continue
        score_lookup[unique_nodes.index(cand.cause), unique_nodes.index(cand.effect)] = cand.gain_est
        candidate_sinks.add(cand.cause)
    if len(candidate_sinks) == 1:
        return candidate_sinks.pop()
    deltas = score_lookup - score_lookup.T
    np.fill_diagonal(deltas, -np.inf)
    # for each node find the best delta 
    optimal_delta_per_node = np.max(deltas, axis=1)
    # return the node with the smallest delta
    return unique_nodes[np.argmin(optimal_delta_per_node)]

    

def find_topological_order(edge_scores):
    '''
    Return topological order given edge scores (data frame from score_all_edges).
    Order is from source first to sink last.
    It is found by taking the node which is predicted the worst by any node after it in the order.
    It is inserted into the order and the process is repeated with the remaining nodes.
    Code to use topological order to rule out edges:
    if topological_order.index(x) > topological_order.index(y):
        # x is after y in the topological order
        do something
    '''
    score_lookup = {}
    for index, row in edge_scores.iterrows():
        score_lookup[(row['x'],row['y'])] = row['opt_save_est'] 
    candidate_sinks = list(edge_scores['x'].unique())

    return __find_topological_order(score_lookup, candidate_sinks)