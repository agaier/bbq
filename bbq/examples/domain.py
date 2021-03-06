import numpy as np

# Test Domain
import math
from bbq.domains._domain import RibsDomain
from bbq.utils import scale
from bbq.parallel import create_dask_client, dask_eval

class Rastrigin(RibsDomain):
    def __init__(self, p):
        RibsDomain.__init__(self, p) 
        self.param_bounds = p['param_bounds'] 
    
    def _fitness(self, x):        
        f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
        return -f + 2*x.shape[0]**2 # Scale to make QD score increasing
    
    def _desc(self, x):
        return np.array(x[0:self.n_desc])

    def express(self, x):
        return scale(x, self.param_bounds)

    def prep_eval(self, p):
        client = create_dask_client(p['n_workers']) 
        return client

    def batch_eval(self, xx, client):
        objs, bcs, pheno = dask_eval(xx, self.evaluate, client) 
        metas = pheno
        return objs, bcs, metas