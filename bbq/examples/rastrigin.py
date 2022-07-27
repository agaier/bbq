import numpy as np

# Test Domain
import math
from bbq.domains._domain import RibsDomain
from bbq.utils import scale
from bbq.parallel import create_dask_client, dask_eval

class Rastrigin(RibsDomain):
    def __init__(self, param_bounds=[-5.12, 5.12], **kwargs):
        self.param_bounds = param_bounds
        RibsDomain.__init__(self, **kwargs)        
    
    def _fitness(self, x):        
        f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
        return -f + 2*x.shape[0]**2 # Scale to make QD score increasing
    
    def _desc(self, x):
        return np.array(x[0:2])

    def express(self, x):
        return scale(x, self.param_bounds)