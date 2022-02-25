from dask.distributed import Client, LocalCluster
import numpy as np

def dask_eval(xx, eval_func, client, serial=False):
    ''' Performs parallel evaluation across dask workers''' 
    objs, descs, phenos = [], [], []           
    if serial:
        for x in xx:
            obj, desc, pheno = eval_func(x)
            objs.append(obj)
            descs.append(desc)
            phenos.append(pheno)
    else:
        futures = client.map(lambda x: eval_func(x), xx)
        results = client.gather(futures)
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