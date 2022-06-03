""" Simple rastrigin example using bbq pattern  """
from bbq.archives.grid_archive import GridArchive_Obj
from bbq.emitters.obj_emitter import ObjEmitter
from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites
from bbq.archives import GridArchive

from bbq.emitters import GaussianEmitter

def run_me(domain, p, emitter_type=GaussianEmitter, archive_type=GridArchive):
    d = domain(p)
    logger = RibsLogger(p)
    archive = map_elites(d, p, logger, emitter_type=emitter_type, 
                                       archive_type=archive_type)    
    logger.zip_results()
    print('\n[*] Done')

if __name__ == '__main__':
    from bbq.examples.domain import Rastrigin
    from bbq.examples.domain_obj import Rastrigin_obj
    from bbq.utils import create_config
    
    base_config = 'config/rast.yaml'
    #exp_config = 'config/test.yaml'
    exp_config = 'config/smoke.yaml'
    
    p = create_config(base_config, exp_config)

    run_me(Rastrigin, p)
    # run_me(Rastrigin_obj, p, emitter_type=ObjEmitter, 
    #                          archive_type=GridArchive_Obj)
