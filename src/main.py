import argparse
import pandas as pd
import numpy as np

from . import dataloader as dl
from . import mdl_based_search
from . import our_gloabls
from . import distributions


# If number of events is not provided, it is assumed to be the number of unique events in the dataset
def cascade(events, number_of_events=None, allow_instant=True, return_alignment=False ,precision=2, max_delay=300):
    """
    Runs the cascade algorithm on the given events.

    Parameters:
    events: pd.DataFrame with columns event, timestamp
    number_of_events: int, optional, if not provided it is assumed to be the number of unique events in the dataset
    allow_instant: bool, optional, if True, instant effects are allowed
    return_alignment: bool, optional, if True, the alignment of the events is returned
    precision: int, optional, precision for universal real encoding
    max_delay: int, optional, max delay between cause and effect

    Returns:
    estimated_graph: np.ndarray, adjacency matrix of the estimated causal graph, size number_of_events x number_of_events
    aligned_alarms: pd.DataFrame,  with the assigned cause of each event.
    """
    # events: pd.DataFrame with columns event, timestamp
    assert "event" in events.columns and "timestamp" in events.columns
    events = events.rename(columns={"event": "alarm_id", "timestamp": "start_timestamp"}, inplace=False)
    events["device_id"] = 0

    assert pd.api.types.is_integer_dtype(events["alarm_id"].dtype) 
    if pd.api.types.is_float_dtype(events["start_timestamp"].dtype):
        if (events["start_timestamp"] % 1 == 0).all(): # check if all timestamps are integers
            events["start_timestamp"] = events["start_timestamp"].astype(int)
        else:
            events, multiplier = dl.transform_from_float(events)
            max_delay = int(max_delay * multiplier) 
            print("Warning: Timestamps have been auto transformed from float to integer. \nIf the minimal difference between timestamps is very small, consider transforming timestamps manually.")

    elif not (pd.api.types.is_integer_dtype(events["start_timestamp"].dtype)):
        raise ValueError("Invalid timestamp type, must be int or float")
    
    # If number of events is not provided, it is assumed to be the number of unique events in the dataset
    if number_of_events is None:
        number_of_events = events["alarm_id"].nunique()
    elif number_of_events < events["alarm_id"].nunique():
        raise ValueError("Number of events must be greater or equal to the number of unique events in the dataset")
    
    causal_prior = np.zeros((number_of_events,number_of_events), dtype=int)
    causal_prior.fill(-1)

    topology_matrix = np.zeros((1,1), dtype=int)

    our_gloabls.precision = precision
    our_gloabls.max_delay = max_delay
    our_gloabls.instant_effects = allow_instant


    m = mdl_based_search.topological_search(events, topology_matrix, causal_prior, init_with_self_loops=False)
    estimated_graph = np.vectorize(lambda x: 0 if x==None else 1)(m.edges)
    
    if return_alignment:
        def get_ids(alarm):
            if alarm[4] != -1 and alarm[5] != -1:
                cause = np.where((m.all_alarms[:,0]==alarm[4]) & (m.all_alarms[:,1] == alarm[5]))[0]
                assert len(cause) == 1
                return cause[0]
            else:
                return -1

        cause_ids = np.apply_along_axis(get_ids, 1, m.all_alarms)
        # np.vectorize(get_ids)(m.all_alarms)
        alarms_with_cause = np.c_[m.all_alarms, cause_ids]
        aligned_alarms = pd.DataFrame(alarms_with_cause)
        aligned_alarms.drop(columns=[0,1,4,5,8], inplace=True)
        
        aligned_alarms.rename(columns={0:"device_id",2:"timestamp",3:"event",6:"delay",7:"cause_event",9:"cause_index"}, inplace=True)
        aligned_alarms = aligned_alarms[["event", "timestamp", "cause_event", "delay", "cause_index"]]
        return estimated_graph, aligned_alarms
    else:
        return estimated_graph

if __name__ == "__main__":

    raise NotImplementedError("CLI deprecated, please use cascade function instead")

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-alarms", required=True, type=str, help="path to alarms csv") #
    argparser.add_argument("-output", required=True, type=str, help="path to output file")
    argparser.add_argument("-topology", type=str, help="path to topology.npy", default=None)
    argparser.add_argument("-causal",  type=str, help="path to causal_prior.npy", default=None)
    argparser.add_argument("-true-graph",  type=str, help="path to rca_prior.npy", default=None) 
    argparser.add_argument("-precision",  type=int, help="precision for universal real encoding", default=our_gloabls.precision)
    argparser.add_argument("-candidate-delay",  type=int, help="max delay for candidate edges", default=our_gloabls.max_delay)
    argparser.add_argument("--no-topo", action="store_true", help="don't use topology", default=False)
    argparser.add_argument("--init-with-self-loops", action="store_true", help="init model with self loops", default=False) 
    argparser.add_argument("--optimize-dst", action="store_true", help="optimize dst", default=our_gloabls.optimize_dst) 
    argparser.add_argument("--float-input", action="store_true", help="use when input is in float format and should be discretized", default=False) 
    argparser.add_argument("-search", type=str, help="search algorithm to use, topo or greedy. To compute length under causal prior use prior", default="topo")
    argparser.add_argument("-align", type=str, help="alignment-method to be used, next or mnnd ", default=our_gloabls.align_mode) 
    argparser.add_argument('-dst', type=str, help="distribution to use", default="geometric")
    argparser.add_argument("--instant", action="store_true", help="allow instant effects", default=our_gloabls.instant_effects)
    argparser.add_argument("--instant-idf", action="store_true", help="allow instant effects", default=our_gloabls.instant_idf) 
    argparser.add_argument("--mle-bic", action="store_true", help="use mle bic", default=our_gloabls.mle_bic) 

    
    
    args = argparser.parse_args()

    print(args)


    our_gloabls.precision = args.precision
    our_gloabls.max_delay = args.candidate_delay
    our_gloabls.optimize_dst = args.optimize_dst
    our_gloabls.align_mode = args.align
    our_gloabls.instant_effects = args.instant
    our_gloabls.instant_idf = args.instant_idf
    our_gloabls.mle_bic = args.mle_bic

    if our_gloabls.instant_idf:
        args.output += ".instant-idf.npy"


    if args.dst == "geometric":
        our_gloabls.distribution = distributions.GeometricDistribution
    elif args.dst == "poisson":
        our_gloabls.distribution = distributions.PoissonDistribution
    elif args.dst == "uniform":
        our_gloabls.distribution = distributions.UniformDistributionCont if our_gloabls.mle_bic else distributions.UniformDistributionDst
    elif args.dst == "normal":
        our_gloabls.distribution = distributions.DiscreteNormalDistribution
    elif args.dst == "exponential":
        our_gloabls.distribution = distributions.ExponentialDistribution
    else:
        raise ValueError("invalid distribution")

    alarms = pd.read_csv(args.alarms)
    if args.float_input:
        alarms = dl.transform_from_float(alarms)
    if args.topology is None:
        topology = get_device_topology.get_topolgy(alarms)
    elif args.topology == "full":
        topology = dl.get_full_connected_topology_matrix(alarms)
    elif args.topology == "empty":
        topology = dl.get_empty_topology_matrix(alarms)
    elif args.topology == "all-in-one":
        # pretend all alarms occur on the same device
        alarms["device_id"] = 0
        topology = dl.get_empty_topology_matrix(alarms)
    else:
        topology = np.load(args.topology)
    if args.causal is None:
        causal_prior = dl.get_empty_causal_prior(alarms, args.true_graph)
    else:
        causal_prior = np.load(args.causal)

    if args.search == "greedy":
        m = mdl_based_search.search(alarms, topology, causal_prior, no_topo=args.no_topo)
    elif args.search == "topo":
        m = mdl_based_search.topological_search(alarms, topology, causal_prior, init_with_self_loops=args.init_with_self_loops)
    elif args.search == "prior": # stupid name change it 
        m = mdl_based_search.set_matrix_as_model(alarms, topology, causal_prior, skip_reassign=True)
        m.set_true_explanation(alarms)
        print(causal_prior)

    else:
        raise ValueError("invalid search algorithm")
    print("Final model Length: ", m.compute_length())
    estimated_graph = np.vectorize(lambda x: 0 if x==None else 1)(m.edges)
    np.save(args.output , estimated_graph) 

    def get_ids(alarm):
        if alarm[4] != -1 and alarm[5] != -1:
            cause = np.where((m.all_alarms[:,0]==alarm[4]) & (m.all_alarms[:,1] == alarm[5]))[0]
            assert len(cause) == 1
            return cause[0]
        else:
            return -1

    cause_ids = np.apply_along_axis(get_ids, 1, m.all_alarms)
    # np.vectorize(get_ids)(m.all_alarms)
    alarms_with_cause = np.c_[m.all_alarms, cause_ids]
    aligned_alarms = pd.DataFrame(alarms_with_cause)
    aligned_alarms.drop(columns=[1,4,5,8], inplace=True)
    aligned_alarms.rename(columns={0:"device_id",2:"start_timestamp",3:"alarm_id",6:"delay",7:"cause_id",9:"cause_index"}, inplace=True)
    aligned_alarms.to_csv(args.output + ".aligned.csv") 