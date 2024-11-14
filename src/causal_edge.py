from __future__ import annotations
import warnings
import math
import numpy as np
np.seterr(divide = 'ignore') 
from scipy.stats import geom

from . import util
from . import our_gloabls
from .distributions import DistributionTemplate
class Edge:
    # edge x -> y
    def __init__(self, x:int, y:int,dst:DistributionTemplate) -> None:
        self.cause = x
        self.effect = y
        self.dst = dst
    
    def __repr__(self) -> str:
        return f"Edge(x={self.cause},y={self.effect},dst={self.dst})"
    

    def get_dealy_cost_vectorized(self, delays:np.ndarray) -> np.ndarray:
        return self.dst.compute_cost_per_delay(delays)

    def get_skip_cost(self) -> float:
        if self.dst.skip_fraction < 0 or self.dst.skip_fraction > 1:
            print(self.dst.skip_fraction)
            raise ValueError("skip_fraction must be between 0 and 1")
        return self.dst.compute_skip_cost(1)
    
    def get_edge_cost(self, N) -> float: 
        # bits = 2 * math.log2(N)
        # bits += self.dst.getParameterCost()
        # return bits

        return self.dst.getParameterCost()



# create costom dtype for edge candidates - once I know how to define greate and less than realations for them
edgeCanddtype = np.dtype([('cause',np.int64),('effect',np.int64),('lambda',np.float64),('gain_est',np.float64)])

class EdgeCandidate:
    def __init__(self, x:int, y:int, dst:DistributionTemplate, gain_est:float) -> None:
        self.cause = x
        self.effect = y
        self.dst = dst
        self.gain_est = gain_est
    
    def __lt__(self, other:EdgeCandidate) -> bool:
        return self.gain_est < other.gain_est
    
    def __gt__(self, other:EdgeCandidate) -> bool:
        return self.gain_est > other.gain_est
    
    def __eq__(self, other:EdgeCandidate) -> bool:
        return self.gain_est == other.gain_est
    
    def __le__(self, other:EdgeCandidate) -> bool:
        return self.gain_est <= other.gain_est
    
    def __ge__(self, other:EdgeCandidate) -> bool:
        return self.gain_est >= other.gain_est
    
    def __repr__(self) -> str:
        return f"EdgeCandidate(x={self.cause},y={self.effect},dst={self.dst},gain_est={self.gain_est})"
    
    def get_edge(self) -> Edge:
        return Edge(self.cause, self.effect, self.dst)