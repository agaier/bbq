""" General MAP-Elites flow using BBQ 
"""
import time

# PyRibs MAP-Elites Framework
from ribs.emitters import IsoLineEmitter
from ribs.optimizers import Optimizer

# BBQ Helpers
from bbq.archives import GridArchive
from bbq.emitters._init_emitters import init_emitters

def map_elites(d, p, logger, 
                    emitter_type=IsoLineEmitter, archive_type=GridArchive):
    # - Setup -----------------------------------------------------------------#
    # : Initial solutions
    start_xx = d.init(p['n_init'])
    evaluator = d.prep_eval(p)                              # evaluation stack
    objs, descs, metas = d.batch_eval(start_xx, evaluator)

    # : Setup emitters and archive
    archive = archive_type(p)      
    emitters = init_emitters(p, archive, start_xx)
    opt = Optimizer(archive, emitters)                      
    archive.add_batch(start_xx, objs, descs, metas)

    # - Main Loop -------------------------------------------------------------#
    non_logging_time = 0.0
    for itr in range(1, p['n_gens']+1):
        itr_start = time.time()       
        # - MAP-ELITES --------------------------------------------------------#
        inds = opt.ask()                                # Create new solutions
        objs, bcs, meta = d.batch_eval(inds, evaluator) # Evaluate solutions
        opt.tell(objs, bcs, meta)                       # Add to archive

        # - Logging -----------------------------------------------------------#
        itr_time = time.time() - itr_start
        non_logging_time += itr_time
        logger.log_metrics(opt, d, itr, itr_time)

    logger.final_log(opt, d, itr, non_logging_time)
    return archive
