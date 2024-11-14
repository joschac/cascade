from scipy.stats import poisson, norm, geom, expon
import math
import numpy as np

from .util import universal_integer_encoding, universal_real_encoding, universal_real_get_encoded_value
from . import our_gloabls as global_vars



################################## Possible Distributions to add ##################################
# Gamma distribution
# Beta distribution
# Pareto distribution
# + Implement Normal and Uniform 

def compute_mean(unique_deltas,unique_delta_counts):
    delta_sum = 0
    for delta, counts in zip(unique_deltas, unique_delta_counts):
        delta_sum += counts * delta
    return 0 if delta_sum == 0 else delta_sum/sum(unique_delta_counts)

def compute_scale(mean, unique_deltas, unique_delta_counts):
    delta_sum = 0
    for delta, counts in zip(unique_deltas, unique_delta_counts):
        delta_sum += counts * (delta - mean)**2
    return 0 if delta_sum == 0 else math.sqrt(delta_sum/sum(unique_delta_counts))

def compute_succ_and_fails(unique_deltas, unique_delta_counts, instant_effects=False):
    instanta_adjustment = 0 if instant_effects else 1
    succ = sum(unique_delta_counts)
    if succ == 0:
        return 0,0
    fails = sum(map(lambda dc: (dc[0]-instanta_adjustment)*dc[1], zip(unique_deltas,unique_delta_counts)))
    assert(fails >= 0)
    return succ, fails

def compute_succ_rate(unique_deltas, unique_delta_counts, instant_effects=False):
    succ, fails = compute_succ_and_fails(unique_deltas, unique_delta_counts, instant_effects=instant_effects)
    return succ/(succ+fails) if succ != 0 else 0


class DistributionTemplate():
    def __init__(self,unique_deltas,unique_delta_counts,skip_fraction):
        assert len(unique_deltas) == 0 or min(unique_deltas) >= 0
        self.unique_deltas = unique_deltas
        self.unique_delta_counts = unique_delta_counts
        self.skip_fraction = skip_fraction #TODO include skip fraction in the parameter cost.
    def __repr__(self) -> str:
        return f"skip_fraction={self.skip_fraction}"
    def compute_cost_per_delay(self, delays):
        raise RuntimeError("Called template class")
    def compute_skip_cost(self, skips):
        if global_vars.mle_bic:
            return skips * -math.log(self.skip_fraction) if skips > 0 and self.skip_fraction > 0 else 0
        else:
            return skips* -math.log2(self.skip_fraction) if skips > 0 and self.skip_fraction > 0 else 0
    def getParameterCost(self):
        if global_vars.mle_bic:
            raise RuntimeError("MLE computes skip cost ")
        return universal_real_encoding(self.skip_fraction, global_vars.precision) if self.skip_fraction > 0 else 0
    def getParameterCount(self):
        raise RecursionError("Called template class")

class PoissonDistribution(DistributionTemplate):
    def __init__(self, unique_deltas,unique_delta_counts, skip_fraction):
        if global_vars.mle_bic:
            raise NotImplementedError("MLE BIC not implemented for Poisson")
        super().__init__(unique_deltas, unique_delta_counts, skip_fraction)
        mean = compute_mean(unique_deltas, unique_delta_counts)
        if mean != 0:
            self.lambdaa = universal_real_get_encoded_value(mean, global_vars.precision)
        else:
            self.lambdaa = 0
    def __repr__(self) -> str:
        return "PoissonDistribution " + super().__repr__() + f" lambda={self.lambdaa}"
    
    def compute_cost_per_delay(self, delays):
        '''
        computes the cost of the given delays  - does not factor in the cost for the skips 
        '''
        delay_cost = -np.log2(poisson.pmf(delays,self.lambdaa)) 
        delay_cost -= np.log2(1-self.skip_fraction) if delays.size > 0 else np.array([]) 
        return delay_cost
        
    def getParameterCost(self):
        cost = super().getParameterCost()
        cost += 0 if self.lambdaa == 0 else universal_real_encoding(self.lambdaa, global_vars.precision)
        return cost


class UniformDistributionDst(DistributionTemplate):
    def __init__(self, unique_deltas, unique_delta_counts, skip_fraction):
        if global_vars.mle_bic:
            raise NotImplementedError("MLE BIC not implemented for Uniform")
        super().__init__(unique_deltas, unique_delta_counts, skip_fraction)
        if len(unique_deltas) != 0:
            self.right = unique_deltas.max()
            self.left = unique_deltas.min()
        else:
            self.right = 0
            self.left = 0
    def __repr__(self) -> str:
        return "UniformDistribution " + super().__repr__() + f" left={self.left} right={self.right}"
    
    def compute_cost_per_delay(self, delays):
        if delays.size == 0:
            return np.array([])
        
        deltas = self.right - self.left + 1
        delay_cost = np.full(delays.size, -math.log2(1/deltas))
        delay_cost -= np.log2(1-self.skip_fraction)
        inf_values_1 = np.argwhere(delays > self.right)
        inf_values_2 = np.argwhere(delays < self.left)
        delay_cost[inf_values_1] = np.inf
        delay_cost[inf_values_2] = np.inf
        return delay_cost


    def getParameterCost(self):
        cost = super().getParameterCost()
        cost += universal_integer_encoding(self.right+1) + math.log2(self.right+1) 
        return cost

class UniformDistributionCont(DistributionTemplate):
    def __init__(self, unique_deltas, unique_delta_counts, skip_fraction): 
        raise NotImplementedError("UniformDistributionCont not implemented yet!")
        super().__init__(unique_deltas, unique_delta_counts, skip_fraction)
        # TODO 

class GeometricDistribution(DistributionTemplate):
    def __init__(self, unique_deltas, unique_delta_counts, skips_fraction):
        if global_vars.mle_bic:
            raise NotImplementedError("MLE BIC not implemented for Geometric")
        super().__init__(unique_deltas, unique_delta_counts, skips_fraction)
        p = compute_succ_rate(unique_deltas, unique_delta_counts, global_vars.instant_effects)
        if p > 0:
            self.p = universal_real_get_encoded_value(p, global_vars.precision)
        else:
            self.p = 0

    def __repr__(self) -> str:
        return "GeometricDistribution " + super().__repr__() + f" p={self.p}"

    def compute_cost_per_delay(self, delays):
        '''
        computes the cost of the given delays  - does not factor in the cost for the skips 
        '''
        delay_cost = -np.log2(geom.pmf(delays,self.p, loc=-1)) if global_vars.instant_effects else -np.log2(geom.pmf(delays,self.p))
        delay_cost -= np.log2(1-self.skip_fraction) if delays.size > 0 else np.array([]) 
        return delay_cost

    def getParameterCost(self):
        cost = super().getParameterCost()
        cost += 0 if self.p == 0 else universal_real_encoding(self.p, global_vars.precision) 
        return cost


class ExponentialDistribution(DistributionTemplate):
    def __init__(self, unique_deltas, unique_delta_counts, skip_fraction):
        if not global_vars.mle_bic:
            raise NotImplementedError("ExponentialDistribution only implemented for MLE BIC")
        super().__init__(unique_deltas, unique_delta_counts, skip_fraction)
        mean = compute_mean(unique_deltas, unique_delta_counts)
        if mean != 0:
            self.lambdaa = 1/mean
        else:
            self.lambdaa = 0
    def __repr__(self) -> str:
        return "ExponentialDistribution " + super().__repr__() + f" lambda={self.lambdaa}"
    def compute_cost_per_delay(self, delays):
        delay_cost = - expon.logpdf(delays, scale=1/self.lambdaa)
        delay_cost -= np.log(1-self.skip_fraction) if delays.size > 0 else np.array([])
        return delay_cost
    
    @staticmethod
    def getParameterCount():
        return 1

class DiscreteNormalDistribution(DistributionTemplate):
    def __init__(self, unique_deltas, unique_delta_counts, skip_fraction):
        if global_vars.mle_bic:
            raise NotImplementedError("MLE BIC not implemented for Discrete Normal use NormalDistribution instead")
        super().__init__(unique_deltas, unique_delta_counts, skip_fraction)
        mean = compute_mean(unique_deltas, unique_delta_counts)
        scale = compute_scale(mean, unique_deltas, unique_delta_counts)
        if mean != 0:
            self.mean = universal_real_get_encoded_value(mean, global_vars.precision)
        else:
            self.mean = 0
        if scale < 0.1: # at this point nearly all probably mass is allocated to one delta, lower really does not make sense
            scale = 0.1
        self.scale = universal_real_get_encoded_value(scale, global_vars.precision)
    
    def __repr__(self) -> str:
        return "NormalDistribution " + super().__repr__() + f" mean={self.mean} scale={self.scale}"
    
    def compute_cost_per_delay(self, delays):
        probability = norm.cdf(delays+0.5, loc=self.mean, scale=self.scale) - norm.cdf(delays-0.5, loc=self.mean, scale=self.scale)
        probability = np.where(probability < 10**(-300), 10**(-300), probability)
        delay_cost = -np.log2(probability)
        delay_cost -= np.log2(1-self.skip_fraction) if delays.size > 0 else np.array([])
        return delay_cost

    def getParameterCost(self):
        cost = super().getParameterCost()
        cost += 0 if self.mean == 0 else universal_real_encoding(self.mean, global_vars.precision)
        cost += universal_real_encoding(self.scale, global_vars.precision)
        return cost
    
class NormalDistribution(DistributionTemplate):
    def __init__(self, unique_deltas, unique_delta_counts, skip_fraction):
        raise NotImplementedError("Normal not implemented yet!")

class NonParametricDistribution(DistributionTemplate):
    def __init__(self, unique_deltas, unique_delta_counts):
        raise NotImplementedError("NonParametricDistribution is not properly implemented - requires different handling when usage update")
    #  we do not want to encode delays that we no longer use - this is less relevant for the other distributions as there we only encode the parameters


        super().__init__(unique_deltas, unique_delta_counts)
        nonzero_values = np.nonzero(unique_delta_counts)
        self.non_zero_unique_deltas = unique_deltas[nonzero_values]
        self.non_zero_unique_delta_counts = unique_delta_counts[nonzero_values]
    
    def compute_delta_cost(self):
        sum_all = np.sum(self.non_zero_unique_delta_counts)
        cost = 0
        for delta,counts in zip(self.non_zero_unique_deltas, self.non_zero_unique_delta_counts):
            if counts > 0:
                cost += counts * -math.log2(counts/sum_all)
        return cost

    def getParameterCost(self):
        if self.non_zero_unique_deltas.size == 0 or self.non_zero_unique_deltas.max() == 0:
                    return 0
        cost = 0
        cost += universal_integer_encoding(self.non_zero_unique_deltas.max())
        cost += math.log2(self.non_zero_unique_deltas.max())
        cost += math.log2(math.comb(self.non_zero_unique_deltas.max(), self.non_zero_unique_deltas.size))
        cost += math.log2(math.comb(np.sum(self.non_zero_unique_delta_counts)-1,self.non_zero_unique_deltas.size-1))
        return cost
    