import numpy as np
import fire
from matplotlib import pyplot as plt

# Test Domain
from bbq.examples.rastrigin import Rastrigin

class RastriginInd():
    def __init__(self, genome):
        self.genome = genome

    def mutate(self, p):
        iso_noise  = np.random.randn(len(self.genome))*p['iso_sigma'][0]
        new_genome = np.clip(self.genome + iso_noise, 0.0, 1.0)
        child = RastriginInd(new_genome)
        return child

class Rastrigin_obj(Rastrigin):
    def __init__(self, p):
        Rastrigin.__init__(self, p)  

    def express(self, xx):
        """ This function turns the raw parameter values that the optimization
        algorithm works on (the genotype) into something that can be properly
        evaluated (the phenotype)
        
        This method must be customized for the object type
        """
        return super().express(xx.genome) # Scale genome instead of raw vector

    def init(self, n_solutions):
        """Generates or loads initial solutions

        Args:
            n_solutions (int): Number of solutions to generate

        Returns:
            [NxM np_array]: Initial solutions
        """
        initial_genomes = np.random.rand(n_solutions, self.n_dof)
        initial_pop = []
        for ind in initial_genomes:
            initial_pop += [RastriginInd(ind)] 
        return initial_pop

# - Test Domain ---------------------------------------------------------------#       
def test_rastrigin():
    rast = Rastrigin_obj(x_scale=[-2,2])

    # Evaluate Intial Population
    xx = rast.init(1000)
    obj, desc = [], []
    for x in xx:
        o, d, p = rast.evaluate(x)
        obj += [o]
        desc += [d]        
    desc = np.vstack(desc)    
    obj = np.vstack(obj)   

    # Plot all evaluated solutions in descriptor space
    fig,ax = plt.subplots(figsize=(4,4),dpi=100)
    ax.scatter(desc[:,0], desc[:,1], s=20, c=obj, alpha=0.2)    

    # Mutate Population
    children = []
    for x in xx:
        children += [x.mutate({'iso_sigma': 0.1})]

    # Evaluate again
    obj, desc = [], []
    for x in children:        
        o, d, p = rast.evaluate(x)
        obj += [o]
        desc += [d]        
    desc = np.vstack(desc)    
    obj = np.vstack(obj)   

    ax.scatter(desc[:,0], desc[:,1], s=20, c=obj, alpha=.9)    
    fig.savefig("rast_test_out.png")
    plt.close()
    print('Done')

if __name__ == '__main__':
    fire.Fire(test_rastrigin)