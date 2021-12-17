import numpy as np
import pandas as pd
from ribs.archives import GridArchive
from dask.distributed import Client, LocalCluster


# - Utility Functions ---------------------------------------------------------#
def scale(x, param_scale):
    """ Scale each column of matrix by mins and maxes defined in vectors"""
    v_min, v_max = param_scale[0], param_scale[1]
    v_range = v_max-v_min
    v_scaled = (v_range * x) + v_min
    return v_scaled    


# - Base Domains --------------------------------------------------------------#
class RibsDomain():
    def __init__(self, n_dof, n_desc, x_scale):
        self.n_dof = n_dof
        self.n_desc = n_desc
        self.x_scale = x_scale     

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
        evaluated (the phenotype) """
        return scale(xx, self.x_scale)

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
        return initial_solutions        

    def batch_eval(self, xx):
        ''' Return objective and descriptor value of batch of solutions '''
        if len(xx.shape) == 1:
            objs, descs, phenos = self.evaluate(xx)
        else:
            phenos = []
            objs, descs = np.full(xx.shape[0],np.nan), \
                        np.full((xx.shape[0],self.n_desc),np.nan)
            for i, pheno in enumerate(xx):
                objs[i], descs[i], pheno = self.evaluate(pheno) 
                phenos.append(pheno)
        return objs, descs, phenos

# - Containers ---------------------------------------------------------------#
class npGridArchive(GridArchive):
    def __init__(self, dims, ranges, seed=None, dtype=np.float64):
        super().__init__(dims, ranges, seed=seed, dtype=dtype)

    def as_numpy(self, include_metadata=False):
        # Create array
        grid_res = [len(a)-1 for a in self.boundaries]
        n_channels = sum([1, self._solution_dim, self._behavior_dim])
        np_archive = np.full(np.r_[grid_res, n_channels], np.nan)

        # Fill array
        # --> TODO: work on higher dim grids
        for elite in self:
            elite_stats = np.r_[elite.obj, elite.beh, elite.sol]
            np_archive[elite.idx[0], elite.idx[1], :] = elite_stats
        if not include_metadata:
            return np_archive

        else:
            meta_archive = np.full(np.r_[grid_res, 1], np.nan, dtype=object)
            for elite in self:
                meta_archive[elite.idx[0], elite.idx[1], :] = [elite.meta]
            return np_archive, meta_archive

    def add_batch(self, xx, objs, descs, meta=None):
        if meta is not None:
            for i in range(len(objs)):
                self.add(xx[i], objs[i], descs[i], metadata=meta[i])
        else:
            for i in range(len(objs)):
                self.add(xx[i], objs[i], descs[i])        


def pandas_to_numpy(archive_file, grid_res=None, desc_bounds=None, dof=None, **kwargs):
    """ Converts saved pandas archive to multidimensional np array
    
    Input: archive_file - filename.pd 
           grid_res     - int tuple             - resolution in each dimension
           desc_bounds  - Descriptor x 2 tuple  - range of each descriptor
           dof          - int                   - number of parameters
    
    Output: n_archive   - numpy array with all values easily sliced
            m_archive   - meta data 

    Syntax:
        archive_file = 'sample_data/3d_archive.pd'
               
        n_archive, m_archive = pandas_to_numpy(
            archive_file, 
            grid_res = (12,24,2),
            desc_bounds = ( (0.0, 6.0), (0.0, 12.0), (0.0, 1.0) ),
            dof = 15
            )

        # -- OR use existing param dict -- #
        config_file = 'sample_data/config_3d_archive.yaml'        
        param_dict = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        n_archive, m_archive = pandas_to_numpy(archive_file, **config_file)

        n_archive.shape # desc1, desc2, desc3, values
        >> (12, 24, 2, 19)
        
        # Values:
        0 : Objective
        1+: Descriptors
        -1: Parameters
    
    """
    # - Load components from pandas file
    if type(archive_file) == str:
        pd_coll = pd.read_pickle(archive_file)
    else:
        pd_coll = archive_file
    x = pd_coll.filter(regex="solution*").to_numpy()
    d = pd_coll.filter(regex="behavior*").to_numpy()
    o = pd_coll.filter(regex="objective*").to_numpy()
    m = pd_coll.filter(regex="metadata*").to_numpy()    
    
    # - Create and fill archive
    archive = GridArchive(grid_res, desc_bounds)
    archive.initialize(dof)
    for i in range(len(x)):
        archive.add(x[i], o[i], d[i], metadata=m[i])
        
    # - Fill empty numpy array with values
    n_channels = len(x[0])+len(o[0])+len(d[0])
    np_archive = np.full(np.r_[grid_res, n_channels], np.nan)
    #meta_archive = np.full(np.r_[grid_res, 1], np.nan, dtype=object)
    
    for elite in archive:
        elite_stats = np.r_[elite.obj, elite.beh, elite.sol]
        np_archive[elite.idx[0], elite.idx[1], :] = elite_stats
        #np_archive[elite.idx[0], elite.idx[1], elite.idx[2], :] = elite_stats
        #meta_archive[elite.idx[0], elite.idx[1], elite.idx[2]] = elite.meta
    return np_archive

# - Emitters ------------------------------------------------------------------#
def create_emitter(emitter_choice, archive, p):
    emitters = [emitter_choice(
        archive,
        np.zeros(p['dof'])+0.5,
        p['mut_strength'],
        bounds=[p['param_bounds']]*p['dof'],
        batch_size=p['n_batch'], # Individuals created per loop
        ) for _ in range(p['n_emitters'])]       
    return emitters     


# - Parallelization -----------------------------------------------------------#
def dask_eval(xx, batch_eval, client, serial=False):
    ''' Performs parallel evaluation across dask workers'''            
    if serial:
        return batch_eval(xx)

    objs, descs, phenos = [], [], []
    futures = client.map(lambda x: batch_eval(x), xx)
    results = client.gather(futures)

    # Organize results
    for obj, desc, pheno in results:
        objs.append(obj)
        descs.append(desc)
        phenos.append(pheno)

    objs = np.hstack(objs)
    descs = np.vstack(descs)   

    return objs, descs, phenos

def create_dask_client(n_workers):
    ''' Creats local cluster of dask workers'''            
    cluster = LocalCluster(
    processes=True,  # Each worker is a process.
    n_workers=n_workers,  # Create this many worker processes.
    threads_per_worker=1,  # Each worker process is single-threaded.
    )
    return Client(cluster)    
