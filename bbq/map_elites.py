""" General MAP-Elites flow using BBQ 
"""
import time

# PyRibs MAP-Elites Framework
from ribs.emitters import IsoLineEmitter
from ribs.optimizers import Optimizer

# BBQ Helpers
from bbq.archives import GridArchive
from bbq.create_emitter import create_emitter

def map_elites(d, p, logger, 
                    emitter_type=IsoLineEmitter, archive_type=GridArchive):
    # - Setup -----------------------------------------------------------------#
    archive = archive_type(p)                               # stores solutions
    emitter = create_emitter(emitter_type, archive, p)      # creates solutions
    opt = Optimizer(archive, emitter)                       # MAP-Elites
    evaluator = d.prep_eval(p)                              # evaluation stack

    # - Bootstrap with initial solutions
    start_xx = d.init(p['n_init'])
    objs, descs, metas = d.batch_eval(start_xx, evaluator)
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
