"""This is a test file for extensions and helper functions for the pyRibs
MAP-Elites framework.

-- Features
    - Configuration by a yaml file
    - A custom grid class that export to numpy
    - A logging class for frequent use cases
"""

import numpy as np
from humanfriendly import format_timespan
import fire

from matplotlib import pyplot as plt

# Test Domain
import math
from ribs_helpers import RibsDomain # Base class for ribs domains

class Rastrigin(RibsDomain):
    def __init__(self, n_dof=10, n_desc=2, x_scale=[-5,5]):
        RibsDomain.__init__(self, n_dof, n_desc, x_scale)  
    
    def _fitness(self, x):        
        f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
        return -f + x.shape[0]**6 # Scale to make QD score increasing
    
    def _desc(self, x):
        return np.array(x[0:self.n_desc])

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