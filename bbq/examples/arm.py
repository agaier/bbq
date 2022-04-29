import numpy as np

# Test Domain
import math
from bbq.domains._domain import RibsDomain
from bbq.parallel import create_dask_client, dask_eval

class Arm(RibsDomain):
    def __init__(self, p, seed=0):
        RibsDomain.__init__(self, p) 
        self.param_bounds = p['param_bounds'] 
        if seed is None:
            self.offset = np.ones(self.n_dof)*0.5
        else:
            self.offset = np.random.rand(self.n_dof)
            print(self.offset)
    
    def _fitness(self, pheno): 
        norm_theta = (pheno+math.pi)/2
        return 1 - np.std(norm_theta)
    
    def _desc(self, pheno):
        c = np.cumsum(pheno)
        x = np.sum(np.cos(c)) / (2. * len(pheno)) + 0.5
        y = np.sum(np.sin(c)) / (2. * len(pheno)) + 0.5
        return np.array((x,y))

    def express(self, x):
        shifted_x = (x*2) - self.offset
        clipped_x = np.clip(shifted_x, 0,1)
        pheno = 2 * math.pi * clipped_x - math.pi
        return pheno

    def init(self, n_solutions):
        """Generates or loads initial solutions

        Args:
            n_solutions (int): Number of solutions to generate

        Returns:
            [NxM np_array]: Initial solutions
        """
        #initial_solutions = np.random.rand(n_solutions, self.n_dof)
        initial_solutions = np.random.randn(n_solutions, self.n_dof)
        return initial_solutions    

    def prep_eval(self, p):
        client = create_dask_client(p['n_workers']) 
        return client

    def batch_eval(self, xx, client):
        objs, bcs, pheno = dask_eval(xx, self.evaluate, client) 
        metas = pheno
        return objs, bcs, metas