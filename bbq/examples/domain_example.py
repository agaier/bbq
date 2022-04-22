import numpy as np
import fire
from matplotlib import pyplot as plt

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

# - Test Domain ---------------------------------------------------------------#       
def test_rastrigin():
    rast = Rastrigin(n_dof=10, x_scale=[-2,2])

    # Evaluate function
    xx = np.random.rand(10000,2)
    obj, desc = [], []
    for x in xx:
        o, d, p = rast.evaluate(x)
        obj += [o]
        desc += [d]
        
    desc = np.vstack(desc)    
    obj = np.vstack(obj)   

    # Plot all evaluated solutions in descriptor space
    fig,ax = plt.subplots(figsize=(4,4),dpi=100)
    ax.scatter(desc[:,0], desc[:,1], s=20, c=obj, alpha=0.5)    
    fig.savefig("rast_test_out.png")
    plt.close()

if __name__ == '__main__':
    fire.Fire(test_rastrigin)