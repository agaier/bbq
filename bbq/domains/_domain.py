import numpy as np
import inspect

# - Base Domains --------------------------------------------------------------#
class RibsDomain():
    def __init__(self, p):
        self.n_dof  = p['n_dof']
        self.n_desc = len(p['desc_bounds'])

    def evaluate(self, x):
            """ Evaluates a single individual, giving an objective and descriptor
            value, along with any metadata to be saved (such as the full phenotype)

            Args:
                x ([numpy array]): Raw parameter values between 0 and 1

            Returns:
                obj, desc, pheno: objective, descriptor, metadata
            """
            pheno = self.express(x)
            obj = self._fitness(pheno)
            desc = self._desc(pheno)
            return obj, desc, pheno

    def express(self, xx):
        """ This function turns the raw parameter values that the optimization
        algorithm works on (the genotype) into something that can be properly
        evaluated (the phenotype) 
        
        If not reimplemented in domain pass on genotype as phenotype.
        """
        return xx

    def _fitness(self, x):        
        raise NotImplementedError   
        
    def _desc(self, x):
        raise NotImplementedError   

    def init(self, n_solutions):
        """Generates or loads initial solutions

        Args:
            n_solutions (int): Number of solutions to generate

        Returns:
            [NxM np_array]: Initial solutions
        """
        initial_solutions = np.random.rand(n_solutions, self.n_dof)
        #initial_solutions = np.ones((n_solutions, self.n_dof))*0.5
        return initial_solutions        

    def prep_eval(self, p):
        """ Prepare evaluation if necessary: 
            - start up dask clients
            - set up file structures for external evaluators
            - or nothing, the results here will be used by batch eval
        """
        return None

    # def batch_eval(self, xx):
    #     ''' Return objective and descriptor value of batch of solutions '''
    #     if is_class(xx):
    #         objs, descs, phenos = self.evaluate(xx)
    #     elif len(xx.shape) == 1:
    #         objs, descs, phenos = self.evaluate(xx)
    #     else:
    #         phenos = []
    #         objs, descs = np.full(xx.shape[0],np.nan), \
    #                     np.full((xx.shape[0],self.n_desc),np.nan)
    #         for i, pheno in enumerate(xx):
    #             objs[i], descs[i], pheno = self.evaluate(pheno) 
    #             phenos.append(pheno)
    #     return objs, descs, phenos

# - Utility Functions ---------------------------------------------------------#

def is_class(o):
    return hasattr(o, '__dict__')