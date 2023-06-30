import numpy as np
from bbq.domains._parallel import create_dask_client, dask_eval


# - Base Domains --------------------------------------------------------------#
class BbqDomain():
    #def __init__(self, n_params=1, n_workers=1, **_):
    def __init__(self, n_dof=1, n_workers=1, **_):
        self.n_dof = n_dof
        self.n_workers = n_workers

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
        initial_solutions = np.random.rand(n_solutions, self.n_params)
        return initial_solutions        

    def prep_eval(self, n_workers=1, **kwargs):
        """ Prepare evaluation if necessary: 
            - start up dask clients
            - set up file structures for external evaluators
            - or nothing, the results here will be used by batch eval
        """
        if n_workers == 1:
            client = None
        else:
            print(f"[*] Starting dask client with {n_workers} workers", end='...')
            client = create_dask_client(n_workers)
            print(f"done.")

        return client

    def batch_eval(self, xx, evaluator=None):
        if evaluator == None:
            objs, descs, metas = zip(*[self.evaluate(x) for x in xx])
        else:
            objs, descs, metas = dask_eval(xx, self.evaluate, evaluator) 
        return objs, descs, metas       


def is_class(o):
    return hasattr(o, '__dict__')