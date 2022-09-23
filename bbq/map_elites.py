""" General MAP-Elites flow using BBQ 
"""
import time
from ribs.optimizers import Optimizer
from bbq.archives._init_archive import init_archive
from bbq.emitters._init_emitter import init_emitter, emitter_lookup


def map_elites(d, p, logger, emitter_lookup=emitter_lookup):
    # - Setup -----------------------------------------------------------------#
    # : Initial solutions
    start_xx = d.init(p['n_init'])
    evaluator = d.prep_eval(**p)   # initialize evaluation stack
    objs, descs, metas = d.batch_eval(start_xx, evaluator)

    # : Setup emitters and archive
    archive = init_archive(p)
    emitter = init_emitter(p, archive, start_xx, emitter_lookup=emitter_lookup)
    opt = Optimizer(archive, emitter)                      
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
