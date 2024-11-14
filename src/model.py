import numpy as np
import pandas as pd
import bisect
import collections
import logging
import math 
from scipy.special import comb
# logging.basicConfig(level=logging.DEBUG)

from . import causal_edge
from . import our_gloabls
from . import alarm
from . import edge_scoring
from . import util
class Model:
    def __init__(self, G, N, total_number_of_alarms, device_dict):
        '''
        G: networkx graph all data from all_data_graph
        N: number of alarm types 
        '''
        self.G = G
        self.edges = np.empty((N,N), dtype=causal_edge.Edge)
        self.device_dict = device_dict
        
        '''
        alarm                                                 explained_by            
        0- device_id, 1- alarm_id, 2- start_time, 3 - type    4- device_id, 5- alarm_id,      6- delay, 7- cause, 8- effect 
        '''
        self.all_alarms = np.full((total_number_of_alarms,9), -1, dtype=np.int32) 
        self.all_alarms_cost = np.zeros(total_number_of_alarms, dtype=np.float32) # cost of each alarm

        self.roll_back = {} # key row_id value: ([device_id, alarm_id, delay, cause, effect],cost) Note: start time and type are not affected 

        # count how often each alarm occurs 
        counts = collections.Counter()
        # for count in map(lambda x: collections.Counter(map(lambda y:y.type,x.alarms)),list(G.nodes())):
        #     counts.update(count)
        for device in G.nodes():
            d_c = collections.Counter(map(lambda y:y.type,device.alarms))
            factor = len(list(G.neighbors(device))) +1 # plus one for self
            # multiple d_c by factor and add to counts
            for k,v in d_c.items():
                counts[k] += v * factor # we can apply a cause edge for each occurrence and each edge
        self.alarm_counts = counts
        # set base cost for each alarm type
        assigned_alarms = 0
        for d in self.G.nodes():
            self.all_alarms[assigned_alarms:assigned_alarms+len(d.alarms),0] = d.id
            self.all_alarms[assigned_alarms:assigned_alarms+len(d.alarms),1] = np.arange(len(d.alarms))
            for i, a in enumerate(d.alarms):
                self.all_alarms[assigned_alarms+i,2:4] = a.start, a.type
            assigned_alarms = assigned_alarms + len(d.alarms)    
        self.all_alarms = self.all_alarms[self.all_alarms[:, 2].argsort()]

        self.T = self.all_alarms[-1,2] - self.all_alarms[0,2]
        self.n_devices = np.unique(self.all_alarms[:,0]).size

        self.base_cost = {}
        self.recompute_base_cost(np.arange(N))
    

    def recompute_base_cost(self, update_list):
        for a in update_list:
            #alarms not explained by any edge
            selection = np.where((self.all_alarms[:,5] == -1) & (self.all_alarms[:,3] == a))
            a_alarms = selection[0].size
            # self.base_cost[a] = math.log2(comb(self.n_devices*self.T+a_alarms-1, a_alarms-1, exact=True))/a_alarms if a_alarms > 0 else 0
            bc = util.BaseCost(self.T, self.n_devices, a_alarms, self.alarm_counts[a])
            self.all_alarms_cost[selection] = bc.cost
            self.base_cost[a] = bc
            

    def set_true_explanation(self, alarms_df:pd.DataFrame):
        self.all_alarms = np.full(self.all_alarms.shape, -1, dtype=np.int32) 
        for i, row in alarms_df.iterrows():
            self.all_alarms[i,0] = row['device_id']
            self.all_alarms[i,1] = i
            self.all_alarms[i,2] = row['start_timestamp']
            self.all_alarms[i,3] = row['alarm_id']
            if row['cause_index'] != -1:
                cause = alarms_df.iloc[row['cause_index']]
                self.all_alarms[i,4] = cause['device_id']
                self.all_alarms[i,5] = row['cause_index']
                self.all_alarms[i,6] = row['start_timestamp'] - cause['start_timestamp']
                self.all_alarms[i,7] = cause['alarm_id']
                self.all_alarms[i,8] = row['alarm_id']
        self.all_alarms = self.all_alarms[self.all_alarms[:, 2].argsort()]
        self.recompute_base_cost(np.arange(self.edges.shape[0]))
        self.__refit_all_edges()

    def test_add_edge(self, edge:causal_edge.Edge) -> list[alarm.Alarm]:
        logging.debug(f"testing edge {edge}")
        old_length = self.__compute_length()
        alarms = self.__add_edge(edge)
        self.recompute_base_cost([edge.effect])
        new_length = self.__compute_length()
        if old_length > new_length:
            logging.debug(f"adding edge {edge}")
            logging.debug(f"old length: {old_length} -> new length: {new_length}")
            self.roll_back = {}
            return alarms
        else:
            self.edges[edge.cause,edge.effect] = None
            logging.debug(f"not adding edge {edge}")
            for a, (values, cost) in self.roll_back.items():
                self.all_alarms[a,4:] = values
                self.all_alarms_cost[a] = cost
            self.recompute_base_cost([edge.effect])
            self.__refit_all_edges()
            return []
    
    def test_remove_effect_edges(self, effect) -> list[alarm.Alarm]: #never called? 
        edge_selections = []
        edges_with_effect = []
        for edge in self.edges.flatten():
            if edge is not None and edge.effect == effect:
                edge_selections.append(np.where((self.all_alarms[:,7] == edge.cause) & (self.all_alarms[:,8] == edge.effect)))
                edges_with_effect.append((edge.cause, edge.effect))
        order = np.argsort([np.average(self.all_alarms_cost[selection]) for selection in edge_selections])[::-1]

        for si in order:
            si_selection = edge_selections[si]
            si_cause, si_effect = edges_with_effect[si]
            if len(si_selection) > 0: 
                # 1. back up edge config
                old_length = self.__compute_length()
                all_alarms_selection_backup = self.all_alarms[si_selection].copy()
                all_alarms_cost_selection_backup = self.all_alarms_cost[si_selection].copy()
                edge_backup = self.edges[si_cause, si_effect]
                # 2. remove edge 
                self.all_alarms[si_selection,4:] = -1
                self.recompute_base_cost([effect])
                # self.all_alarms_cost[si_selection] = self.base_cost[effect]
                self.edges[si_cause, si_effect] = None
                # 3. reassign alarms
                self.__reassing_alarms(si_selection[0])
                # 3.1 refit all edges?? 
                self.__refit_all_edges()
                # 4. recompute length
                new_length = self.__compute_length()
                # 5. if length is better keep edge removed else restore edge config
                if new_length > old_length:
                    self.all_alarms[si_selection] = all_alarms_selection_backup
                    self.all_alarms_cost[si_selection] = all_alarms_cost_selection_backup
                    self.edges[si_cause, si_effect] = edge_backup
                    self.__refit_all_edges()
                else:
                    logging.debug(f"removed edge {si_cause} {si_effect}")
                    logging.debug(f"old length: {old_length} -> new length: {new_length}")

            else:   
                self.edges[si_cause, si_effect] = None

        self.recompute_base_cost([effect])
    
    def remove_effect_edges(self, effect):
        removed_edges = []
        edge_selections = []
        edges_with_effect = []
        for edge in self.edges.flatten():
            if edge is not None and edge.effect == effect:
                edge_selections.append(np.where((self.all_alarms[:,7] == edge.cause) & (self.all_alarms[:,8] == edge.effect)))
                edges_with_effect.append((edge.cause, edge.effect))
        order = np.argsort([np.average(self.all_alarms_cost[selection]) for selection in edge_selections])[::-1]

        for si in order:
            si_selection = edge_selections[si]
            si_cause, si_effect = edges_with_effect[si]
            if len(si_selection) > 0: 
                # 1. back up edge config
                # old_length = self.__compute_length()
                # all_alarms_selection_backup = self.all_alarms[si_selection].copy()
                # all_alarms_cost_selection_backup = self.all_alarms_cost[si_selection].copy()
                # edge_backup = self.edges[si_cause, si_effect]
                # 2. remove edge 
                self.all_alarms[si_selection,4:] = -1
                self.all_alarms_cost[si_selection] = self.base_cost[effect].cost
                self.edges[si_cause, si_effect] = None
                # 3. reassign alarms
                self.__reassing_alarms(si_selection[0])
                # 3.1 refit all edges?? 
                self.__refit_all_edges()
                # 4. recompute length
                removed_edges.append(si_cause)
                logging.debug(f"removed edge {si_cause} {si_effect}")

            else:   
                self.edges[si_cause, si_effect] = None
        self.recompute_base_cost([effect])
        return np.array(removed_edges)

    def __compute_topological_order(self):
        binary_G  = np.vectorize(lambda x: 0 if x is None else 1)(self.edges)
        topological_order = [] 
        while len(topological_order) < binary_G.shape[0]:
            next_roots = np.where(np.sum(binary_G, axis=0) == 0)[0]
            next_roots = list(filter(lambda x: x not in topological_order, next_roots))
            topological_order += next_roots
            binary_G[next_roots] = 0
            
            
        return topological_order
    
    def __get_log_likelihood(self):
        log_likelihood = np.sum(self.all_alarms_cost)
        for edge in self.edges.flatten():
            if edge != None:
                fires = len(self.all_alarms[(self.all_alarms[:,7] == edge.cause) & (self.all_alarms[:,8] == edge.effect)])
                skips = self.alarm_counts[edge.cause] - fires
                assert skips >= 0
                log_likelihood += skips * self.edges[edge.cause,edge.effect].get_skip_cost() if skips != 0 else 0
        return log_likelihood

    def __get_parameter_count (self):
        return (our_gloabls.distribution.getParameterCount()+2 + 1) * np.sum(self.edges != None) # +2 for which notes are connected + 1 for skip fraction
    

    
    def compute_bic_difference(self, likelihood_diff:float, k_diff:int) -> float:
        log_likelihood = self.__get_log_likelihood() + likelihood_diff
        k =  self.__get_parameter_count() + k_diff
        return util.bic(log_likelihood, k, self.all_alarms.shape[0])

    def __compute_bic(self) -> float:
        log_likelihood = self.__get_log_likelihood()
        k = self.__get_parameter_count()
        return util.bic(log_likelihood, k, self.all_alarms.shape[0])


    def __compute_length(self) -> float:
        if our_gloabls.mle_bic:
            return self.__compute_bic()

        # go over all devices and take sum over all alarms 
        alarms_cost = np.sum(self.all_alarms_cost)
        # count how often each causal edge is used
        
        # get all edges from self.edges
        model_cost = 0
        skip_cost = 0
        for edge in self.edges.flatten():
            if edge != None:
                model_cost += edge.get_edge_cost(len(self.G.nodes()))  if edge.cause != edge.effect else 0
                #skip cost per edge 
                fires = len(self.all_alarms[(self.all_alarms[:,7] == edge.cause) & (self.all_alarms[:,8] == edge.effect)])
                skips = self.alarm_counts[edge.cause] - fires
                assert skips >= 0
                skip_cost += skips * self.edges[edge.cause,edge.effect].get_skip_cost() if skips != 0 else 0

        binary_G  = np.vectorize(lambda x: 0 if x is None else 1)(self.edges)
        parents = np.sum(binary_G, axis=0)
        for k,i in enumerate(self.__compute_topological_order()):
            if k > 0:
                model_cost += math.log2(k)
                model_cost += math.log2(math.comb(k,parents[i])) if parents[i] > 0 else 0

        
        return alarms_cost + skip_cost + model_cost

    def compute_length(self) -> float:
        return self.__compute_length()
        
    
    def check_if_cause_avialiable(self, pc:int, a:int, cause:int, effect:int)->bool:
        explaining_alarm_d = self.all_alarms[pc,0]
        explaining_alarm_a = self.all_alarms[pc,1]
        alarm_d = self.all_alarms[a,0]
        return len(np.where((self.all_alarms[:,0] == alarm_d) & (self.all_alarms[:,4] == explaining_alarm_d) & (self.all_alarms[:,5] == explaining_alarm_a) & (self.all_alarms[:,7] == cause) & (self.all_alarms[:,8] == effect))[0]) == 0

    # @staticmethod
    # def get_alarms_by_type(alarms:list[alarm.Alarm], type:int) -> list[alarm.Alarm]:
    #     return [a for a in alarms if a.type == type] #can we use np.where or something
    
    def get_prev_alarms(self, device_id:int, timepoint:int, type:int):
        max_delay:int = our_gloabls.max_delay
        if our_gloabls.instant_effects:
            return np.where((self.all_alarms[:,0] == device_id) & (self.all_alarms[:,2] <= timepoint) & (self.all_alarms[:,2] > timepoint-max_delay) & (self.all_alarms[:,3] == type))[0]
        else:
            return np.where((self.all_alarms[:,0] == device_id) & (self.all_alarms[:,2] < timepoint) & (self.all_alarms[:,2] > timepoint-max_delay) & (self.all_alarms[:,3] == type))[0]
    
    def get_alarms_by_type(self, device_id:int, type:int):
        return np.where((self.all_alarms[:,0] == device_id) & (self.all_alarms[:,3] == type))[0]
    
    def __refit_all_edges(self, effect:int = None):
        for edge in self.edges.flatten():
            if edge != None:
                if effect == None or edge.effect != effect:
                    affected_alarms = np.where((self.all_alarms[:,7] == edge.cause) & (self.all_alarms[:,8] == edge.effect))
                    if len(affected_alarms[0]) == 0:
                        continue
                    c = len(affected_alarms[0])
                    skips = self.alarm_counts[edge.cause] - c
                    # refit distribution
                    skip_fraction = skips/ self.alarm_counts[edge.cause]
                    delays = self.all_alarms[affected_alarms][:,6]
                    values, counts = np.unique(delays, return_counts=True)
                    edge.dst = our_gloabls.distribution(values,counts,skip_fraction) 
                    # reset cost
                    self.all_alarms_cost[affected_alarms] = edge.get_dealy_cost_vectorized(delays)
                    self.recompute_base_cost([edge.effect])

    def __add_edge(self, edge:causal_edge.Edge) -> list[alarm.Alarm]:

        changed_alarms = set()

        for d in self.G.nodes:
            neighbours = list(self.G.neighbors(d)) + [d] # neighbours + self
            
            for a in self.get_alarms_by_type(d.id, edge.effect):
                # a is alarm we want to explain 
                explantion_changed = False
                self.roll_back[a] = (self.all_alarms[a,4:], self.all_alarms_cost[a])
                for n in neighbours:
                    potential_causes = self.get_prev_alarms(n.id, self.all_alarms[a,2], edge.cause)
                    alarm_delays = -self.all_alarms[potential_causes,2] + self.all_alarms[a,2]
                    delays_costs = edge.get_dealy_cost_vectorized(alarm_delays)
                    for i in np.argsort(delays_costs):
                        pc = potential_causes[i]
                        d_cost = delays_costs[i]
                        if d_cost < self.all_alarms_cost[a]:
                            if self.check_if_cause_avialiable(pc, a, edge.cause, edge.effect):
                                self.all_alarms[a,4:] = n.id, self.all_alarms[pc,1], alarm_delays[i], edge.cause, edge.effect 
                                self.all_alarms_cost[a] = d_cost
                                explantion_changed = True
                                break
                        else:
                            break

                if explantion_changed:
                    changed_alarms.add(a)
        
        

        self.edges[edge.cause,edge.effect] = edge
        self.__refit_all_edges(edge.effect)


        selection = np.where(self.all_alarms[:,3] == edge.effect)[0]

        rassigned_alrams = self.__reassing_alarms(selection)
        changed_alarms = changed_alarms.union(rassigned_alrams)
        self.__refit_all_edges(edge.effect)
        while len(rassigned_alrams) > 0:
            rassigned_alrams = self.__reassing_alarms(selection)
            changed_alarms = changed_alarms.union(rassigned_alrams)
            self.__refit_all_edges(edge.effect)
        

        return changed_alarms



    def __reassing_alarms(self, selection) -> set[int]:
        reassigned = set()
        for a in selection:
            d = self.device_dict[self.all_alarms[a,0]]
            neighbours = list(self.G.neighbors(d)) + [d]
            for edge in self.edges.flatten():
                if edge != None and edge.effect == self.all_alarms[a,3]:
                    # copy pasted from __add_edge
                    for n in neighbours:
                        potential_causes = self.get_prev_alarms(n.id, self.all_alarms[a,2], edge.cause)
                        alarm_delays = -self.all_alarms[potential_causes,2] + self.all_alarms[a,2]
                        delays_costs = edge.get_dealy_cost_vectorized(alarm_delays)
                        for i in np.argsort(delays_costs):
                            pc = potential_causes[i]
                            d_cost = delays_costs[i]
                            if d_cost < self.all_alarms_cost[a]:
                                if self.check_if_cause_avialiable(pc, a, edge.cause, edge.effect):
                                    self.all_alarms[a,4:] = n.id, self.all_alarms[pc,1], alarm_delays[i], edge.cause, edge.effect 
                                    self.all_alarms_cost[a] = d_cost
                                    reassigned.add(a)
                                    break
                            else:
                                break
        return reassigned
    def reassign_all_alrams_and_refit(self):
        all_alarms_selecteed = np.arange(self.all_alarms.shape[0])
        self.__reassing_alarms(all_alarms_selecteed)
        self.__refit_all_edges()