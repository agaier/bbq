""" Simple rastrigin example using bbq pattern  """

from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites
from bbq.archives import GridArchive
from bbq.emitters import GaussianEmitter, IsoLineEmitter, ImprovementEmitter



def run_me(domain, p, emitter_type=GaussianEmitter, archive_type=GridArchive):
    d = domain(p)
    logger = RibsLogger(p)
    archive = map_elites(d, p, logger, emitter_type=emitter_type, 
                                       archive_type=archive_type)    
    print('\n[*] Done')

if __name__ == '__main__':
    config_dir = '../config/'
    from bbq.utils import create_config, load_config
    exp_config  = config_dir+'smoke.yaml'    

    # Test Domains
    
    # - Rastrigin    
    from bbq.examples.rastrigin import Rastrigin
    base_config = config_dir+'rast.yaml'
    p = load_config([base_config, exp_config])
    logger = RibsLogger(p)
    domain = Rastrigin(**p)
    archive = map_elites(domain, p, logger)

    # - Planar Arm
    from bbq.examples.planar_arm import PlanarArm  
    base_config = config_dir+'arm.yaml'
    p = load_config([base_config, exp_config])
    logger = RibsLogger(p)
    domain = PlanarArm(**p)
    archive = map_elites(domain, p, logger)

    # - Planar Arm [Line w/Improvement]
    from bbq.examples.planar_arm import PlanarArm  
    emitter_config = config_dir+'line_cma_mix.yaml'
    p = load_config([base_config, exp_config, emitter_config])

    logger = RibsLogger(p)
    domain = PlanarArm(**p)
    archive = map_elites(domain, p, logger)    