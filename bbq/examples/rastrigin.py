import math
import numpy as np
from bbq.domains._domain import BbqDomain
from bbq.utils import scale

class Rastrigin(BbqDomain):
    def __init__(self, param_bounds=[-5.12, 5.12], **kwargs):
        self.param_bounds = param_bounds
        BbqDomain.__init__(self, **kwargs)        
    
    def _fitness(self, x):        
        f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
        return -f + 2*x.shape[0]**2 # Shift to make QD score increasing
    
    def _desc(self, x):
        return np.array(x[0:2])

    def express(self, x):
        return scale(x, self.param_bounds)

"""Example using objects as genomes.

- Method for generating new individuals is contained within object (mutate)
- The 'ObjEmitter' creates new solutions by calling this mutate function
"""

class RastriginInd():
    def __init__(self, genome):
        self.genome = genome

    def mutate(self, p):
        iso_noise  = np.random.randn(len(self.genome))*p['iso_sigma']
        new_genome = np.clip(self.genome + iso_noise, 0.0, 1.0)
        child = RastriginInd(new_genome)
        return child

class Rastrigin_Obj(Rastrigin):
    def __init__(self, **kwargs):
        self.n_dof = kwargs['n_dof']
        Rastrigin.__init__(self, **kwargs)

    def express(self, x):
        return super().express(x.genome) # Evaluate values inside of class
    
    def init(self, n_solutions):
        initial_genomes = np.random.rand(n_solutions, self.n_dof)
        initial_pop = []
        for ind in initial_genomes:
            initial_pop += [RastriginInd(ind)] 
        return initial_pop              
