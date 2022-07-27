import numpy as np

# Test Domain
import math
from bbq.domains._domain import RibsDomain
from bbq.utils import scale
from bbq.parallel import create_dask_client, dask_eval

class PlanarArm(RibsDomain):
    def __init__(self, param_bounds=[0, 1], **kwargs):
        self.param_bounds = param_bounds
        RibsDomain.__init__(self, **kwargs)        
    
    def _fitness(self, x):        
        return 1 - np.std(x)
    
    def _desc(self, thetas):
        c = np.cumsum(thetas)
        x = np.sum(np.cos(c)) / (2. * len(thetas)) + 0.5
        y = np.sum(np.sin(c)) / (2. * len(thetas)) + 0.5
        return np.array((x,y))

    def express(self, x):
        pheno = scale(x, [-math.pi, math.pi])
        return pheno

    def evaluate(self, x):
        """ Evaluates a single individual, giving an objective and descriptor
        value, along with any metadata to be saved (such as the full phenotype)

        Args:
            x ([numpy array]): Raw parameter values between 0 and 1

        Returns:
            obj, desc, pheno: objective, descriptor, metadata
        """
        pheno = self.express(x)
        obj = self._fitness(x) # fitness based on [0:1] variance
        desc = self._desc(pheno)
        return obj, desc, pheno        