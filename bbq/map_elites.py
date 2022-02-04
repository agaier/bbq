"""Example usage of using PyRibs framework with helper rools in ribs_helpers

"""
import time

# PyRibs MAP-Elites Framework
from ribs.emitters import IsoLineEmitter
from ribs.optimizers import Optimizer

# PyRibs Helpers
from bbq.parallel import create_dask_client, dask_eval
from bbq.archives import GridArchive
from bbq.create_emitter import create_emitter
#from ribs.archives import GridArchive


def map_elites(d, p, logger, 
                    emitter_type=IsoLineEmitter, archive_type=GridArchive):
    # Setup
    archive = archive_type(p['grid_res'], p['desc_bounds']) # stores solutions
    emitter = create_emitter(emitter_type, archive, p)      # creates solutions
    opt = Optimizer(archive, emitter)                       # MAP-Elites
    client = create_dask_client(p['n_workers']) 

    # Bootstrap with initial solutions
    start_xx = d.init(p['n_init'])
    objs, descs, metas = dask_eval(start_xx, d.evaluate, client)
    archive.add_batch(start_xx, objs, descs, metas)

    # - Main Loop -------------------------------------------------------------#
    non_logging_time = 0.0
    for itr in range(1, p['n_gens']+1):
        itr_start = time.time()       
        # - MAP-ELITES --------------------------------------------------------#
        inds = opt.ask()
        objs, bcs, pheno = dask_eval(inds, d.evaluate, client) 
        opt.tell(objs, bcs, pheno)

        # - Logging -----------------------------------------------------------#
        non_logging_time += time.time() - itr_start
        logger.log_metrics(archive, itr)

    return archive
