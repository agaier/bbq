""" Tests and examples of BBQ pattern  """

from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites

if __name__ == '__main__':
    config_dir = '../config/'
    from bbq.utils import load_config
    exp_config  = config_dir+'smoke.yaml'    

    # -- Test Domains ----------------------------------------------------- -- #
    # Rastrigin    
    print("\n[*] Rastrigin on Grid w/ Gaussian Emitter")
    from bbq.examples.rastrigin import Rastrigin
    base_config = config_dir+'rast.yaml'
    p = load_config([base_config, exp_config])
    logger = RibsLogger(p)
    domain = Rastrigin(**p)
    archive = map_elites(domain, p, logger)

    # - Rastrigin with Object Genome
    print("\n[*] Rastrigin on Grid as Individual Object w/ Gaussian Mutation")
    from bbq.examples.rastrigin import Rastrigin_Obj
    base_config = config_dir+'rast_obj.yaml'
    p = load_config([base_config, exp_config])
    logger = RibsLogger(p)
    domain = Rastrigin_Obj(**p)
    archive = map_elites(domain, p, logger)    

    # - Planar Arm
    print("\n[*] Planar Arm on CVT w/Improvement Emitters")
    from bbq.examples.planar_arm import PlanarArm  
    base_config = config_dir+'arm.yaml'
    p = load_config([base_config, exp_config])
    logger = RibsLogger(p)
    domain = PlanarArm(**p)
    archive = map_elites(domain, p, logger)

    # - Planar Arm [Line w/Improvement]
    print("\n[*] Planar Arm on CVT w/Line and Improvement Emitters")
    from bbq.examples.planar_arm import PlanarArm  
    emitter_config = config_dir+'line_cma_mix.yaml'
    p = load_config([base_config, exp_config, emitter_config])
    p['exp_name'] = 'mixed_emitter'
    p['n_workers'] = 2 # To test parallel
    logger = RibsLogger(p)
    domain = PlanarArm(**p)
    archive = map_elites(domain, p, logger)    