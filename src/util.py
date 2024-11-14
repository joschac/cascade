import pandas as pd
import math
from scipy.special import comb
from . import our_gloabls as globals

def get_base_alarm_cost(df):
    if globals.mle_bic:
        raise NotImplementedError('BUG - should not be called for MLE BIC')
    # replace base cost with choose over number of alarms per event
    max_start = df['start_timestamp'].max()
    min_start = df['start_timestamp'].min()
    total_time = max_start - min_start
    alram_ids = df['alarm_id'].unique()
    devices = max(len(df['device_id'].unique()),1)
    base_cost = {}
    for a in alram_ids:
        a_alarms = len(df[df['alarm_id'] == a])
        base_cost[a] = math.log2(comb(devices*total_time+a_alarms-1, a_alarms-1, exact=True))/a_alarms if a_alarms > 0 else 0
        # base_cost[a] = math.log2(total_time)
    return base_cost

def universal_integer_encoding(i,c=2.865064):
    if globals.mle_bic:
        raise NotImplementedError('BUG - should not be called for MLE BIC')
    bits = math.log2(c)
    while i>1:
        i = math.log2(i)
        bits += i
    return bits

def universal_real_encoding(z,p):
    if globals.mle_bic:
        raise NotImplementedError('BUG - should not be called for MLE BIC')
    try: #TODO fix this hack
        s = math.ceil(math.log10((10**p)/z)) #TODO WHY LOG 10?
    except ZeroDivisionError:
        s = 1
    return universal_integer_encoding(s) + universal_integer_encoding(math.ceil(z*(10**s)))

def universal_real_get_encoded_value(z,p):
    if globals.mle_bic:
        raise NotImplementedError('BUG - should not be called for MLE BIC')
    s = math.ceil(math.log10((10**p)/z))
    return math.ceil(z*(10**s))/(10**s)

def bic(n_log_l, k, n):
    return 2 * n_log_l + k * math.log(n)


class BaseCost:
    def __init__(self, T, D, alarms, total_alarms): # don't really like it, the only thing that changes are the number of alarms 
        self.T = T
        self.D = D
        self.alarms = alarms
        self.total_alarms = total_alarms
        self.cost = self.computeTotalCostForN(alarms)/alarms if alarms > 0 else 0
    
    def computeTotalCostForN(self,n):
        if n > self.D * self.T and not globals.mle_bic:
            raise ValueError('Number of alarms cannot be greater than D*T')
        
        if globals.mle_bic:
            cost = n * math.log(self.total_alarms/(self.D*self.T))
        
        if globals.instant_idf:
            p = self.total_alarms / (self.D*self.T)
            cost = n * ((-(1-p)*math.log2(1-p) - p*math.log2(p))/p) if n > 0 else 0
            skip_p = (self.total_alarms - n) / self.total_alarms
            cost += self.total_alarms * ((-(1-skip_p)*math.log2(1-skip_p) - skip_p*math.log2(skip_p))) if skip_p > 0 and skip_p < 1 else 0
            
            # cost += universal_integer_encoding(n+1) if n > 0 else universal_integer_encoding(1)
        else:
            p = n / (self.D*self.T)
            cost = n * ((-(1-p)*math.log2(1-p) - p*math.log2(p))/p) if n > 0 else 0
            # cost += universal_integer_encoding(n+1) if n > 0 else universal_integer_encoding(1)


        return cost