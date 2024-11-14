import math
import numpy as np
from nptyping import NDArray
from typing import Any, Tuple
from scipy.stats import geom
from scipy.special import comb
import copy


from . import util
from . import our_gloabls
from .causal_edge import Edge

def align_next(just_delays): # -> NDArray[(Any,), np.int64]:
    return np.array(list(map( lambda x:np.sort(x)[0],just_delays)), dtype=np.int64)

def get_Deltas(data,indices):
    deltas = np.empty(indices.size,dtype=np.int64)
    for i in range(0,indices.size):
        deltas[i] = data[i][indices[i]]
    return deltas

def align_mnnd(just_delays): # -> NDArray[(Any,), np.int64]:
    import pymnnd # this import is here because it will fail on most systems 

    just_delays = list(map(lambda x: np.sort(np.array(x, dtype=np.int64)), just_delays))

    if (len(just_delays) == 0):
        return []

    res = pymnnd.MNND(just_delays, pymnnd.L2) # mnnd_modes = {"L1":pymnnd.L1, "L2":pymnnd.L2, "LINF":pymnnd.LINF}
    mnnd_indices = np.array(pymnnd.nearest_neighbour_indices(just_delays,res.x_best),dtype=np.int64)
    return get_Deltas(just_delays,mnnd_indices)



def compute_opt_save_est(delays, values, counts, base_cost, total_fires, m, cause, effect, ignore_skips = False):
    skip_fraction = (total_fires - sum(counts))/total_fires if not ignore_skips else 0
    dst = our_gloabls.distribution(values, counts, skip_fraction)
    delay_cost = dst.compute_cost_per_delay(delays)
    # remove all delay cost greater than base cost
    delays = delays[delay_cost < base_cost.cost]
    values, counts = np.unique(delays, return_counts=True)

    skip_fraction = (total_fires - sum(counts))/total_fires if not ignore_skips else 0
    dst = our_gloabls.distribution(values, counts, skip_fraction)
    delay_cost = dst.compute_cost_per_delay(delays)
    skip_cost = dst.compute_skip_cost(total_fires - sum(counts))

    prev_alarms = base_cost.alarms
    new_alarms = base_cost.alarms - len(delay_cost)
    new_r_cost = base_cost.computeTotalCostForN(new_alarms)

    opt_save_est = skip_cost + sum(delay_cost) + new_r_cost - prev_alarms * base_cost.cost
    assert skip_fraction >= 0
    return dst, opt_save_est 


def get_rule_candidate(delay_sets, base_cost, total_fires, m, cause, effect, ignore_skips = False):
    if cause == effect:
        return our_gloabls.distribution([],[],0), 0
    optimize_dst = our_gloabls.optimize_dst
    # test alignment with next following
    # just_delays = list(map(lambda x: x[0], delay_sets))
    just_delays = delay_sets
    delays = align_next(just_delays) if our_gloabls.align_mode == 'next' else align_mnnd(just_delays)
    if len(delays) == 0:
        return our_gloabls.distribution([],[],0), 0 # lamb, skip_fraction, opt_save_est
    values, counts = np.unique(delays, return_counts=True)
    dst, opt_save_est = compute_opt_save_est(delays, values, counts, base_cost, total_fires, m, cause, effect, ignore_skips= ignore_skips)
    
    
    if not optimize_dst:
        return dst, opt_save_est

    results = [(dst, opt_save_est)]

    while len(values) > 1:
        best_i = -1 
        best_est = np.inf
        best_dst = None
        for i in range(len(values)):
            new_values = np.delete(values, i)
            new_counts = np.delete(counts, i)
            new_delays = np.array([d for d in delays if d in new_values])
            new_dst, new_opt_save_est = compute_opt_save_est(new_delays, new_values, new_counts, base_cost, total_fires, ignore_skips)
            if new_opt_save_est < best_est:
                best_est = new_opt_save_est
                best_dst = new_dst
                best_i = i
        
        results.append((best_dst, best_est))
        values = np.delete(values, best_i)
        counts = np.delete(counts, best_i)
        
    return min(results, key=lambda x: x[1])
