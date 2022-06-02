import numpy as np

# Test Domain
import math
from bbq.domains._domain import RibsDomain
from bbq.parallel import create_dask_client, dask_eval

class Arm(RibsDomain):
    """Unbiased Planar Arm

    Planar Arm with random offsets, set at instantiation. Clipped 'dead zones' 
    are inserted at random points in the top and bottom of range in an effort
    to remove biases from initialization. 
    
    See planar_arm.ipynb for details and illustration.

    """
    def __init__(self, p, seed=0, slope=2.0):
        RibsDomain.__init__(self, p) 
        self.slope = slope # 1: 100% alive, 1.25: 80% alive, 1.5: 66% alive, 2: 50% alive
        self.param_bounds = p['param_bounds'] 
        if seed is None:
            self.offset = np.ones(self.n_dof)*0.5
        else:
            self.offset = np.random.rand(self.n_dof)
            print(self.offset)
    
    def _fitness(self, pheno): 
        norm_theta = (pheno+math.pi)/(2*math.pi)
        return 1 - np.std(norm_theta)
    
    def _desc(self, pheno):
        c = np.cumsum(pheno)
        x = np.sum(np.cos(c)) / (2. * len(pheno)) + 0.5
        y = np.sum(np.sin(c)) / (2. * len(pheno)) + 0.5
        return np.array((x,y))

    def express(self, x):
        x = (x*self.slope) - (self.offset*(self.slope-1))
        x = np.clip(x,0,1)
        pheno = 2 * math.pi * x - math.pi    
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