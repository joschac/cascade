import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from . import edge_scoring
from . import util

from .dataloader import read_alarm_data, get_alarms_df
from .get_delays import get_event_delays
from . import causal_edge

def score_all_edges(delays:pd.DataFrame, base_cost, alarm_counts,m): #->np.ndarray[causal_edge.EdgeCandidate]:
    def apply_to_delays(x):
        cause = x['cause']
        effect = x['effect']
        dst, opt_save_est = edge_scoring.get_rule_candidate(x['events delays'],base_cost[effect], alarm_counts[cause],m,cause, effect)
        return causal_edge.EdgeCandidate(cause,effect,dst, opt_save_est)
        # for when using cutome dtype
        # return (x['cause'],x['effect'],lamb, opt_save_est)
    return delays.apply(apply_to_delays, axis=1).to_numpy()



if __name__ == '__main__':
    raise DeprecationWarning("This part of the code is deprecated. Use score_all_edges function instead.")