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
    def __init__(self, p, slope=1.5, seed=None, uniform_init=True):
        RibsDomain.__init__(self, p) 
        self.uniform_init = uniform_init
        self.slope = slope # 1: 100% active, 1.5: 66% active, 2: 50% active
        self.param_bounds = p['param_bounds'] 
        if seed is not None:
            np.random.seed(seed)            
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
        if self.uniform_init:
            initial_solutions = np.random.rand(n_solutions, self.n_dof)
        else:
            # Normal distributed solutions between 0 and 1
            rand_start = 0.5+np.random.randn(n_solutions, self.n_dof)*0.2
            initial_solutions = np.clip(rand_start,0,1)
        return initial_solutions    

    def prep_eval(self, p):
        client = None
        return client

    def batch_eval(self, xx, client=None):
        objs, descs, pheno = zip(*[self.evaluate(x) for x in xx])
        return objs, descs, pheno