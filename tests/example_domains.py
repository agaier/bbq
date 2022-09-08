""" Tests and examples of BBQ pattern  """

from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites

if __name__ == '__main__':
    config_dir = '../config/'
    #config_dir = 'config/'
    from bbq.utils import load_config
    exp_config  = config_dir+'x_smoke.yaml'    
    exp_config  = config_dir+'x_test.yaml'    
    
    arm_config  = config_dir+'d_arm.yaml'    
    rast_config = config_dir+'d_rast.yaml'    
    rast_obj_config = config_dir+'d_rast_obj.yaml' # objects instead of vectors

    gaus_config  = config_dir+'e_gauss.yaml'    
    line_config  = config_dir+'e_line.yaml'    
    cmame_config = config_dir+'e_cmame.yaml'    
    mixed_config = config_dir+'e_mixed.yaml'    


    # -- Test Domains ----------------------------------------------------- -- #
    # - Rastrigin with Gaussian
    print("\n[*] Rastrigin on Grid w/ Gaussian Emitter")
    from bbq.examples.rastrigin import Rastrigin
    p = load_config([rast_config, exp_config, gaus_config])
    logger = RibsLogger(p)
    domain = Rastrigin(**p)
    archive = map_elites(domain, p, logger)

    # - Rastrigin with Object Genome
    print("\n[*] Rastrigin on Grid as Individual Object w/ Gaussian Mutation")
    from bbq.examples.rastrigin import Rastrigin_Obj
    p = load_config([rast_obj_config, exp_config]) # TODO: object emitter!
    p['exp_name'] = 'object'
    logger = RibsLogger(p)
    domain = Rastrigin_Obj(**p)
    archive = map_elites(domain, p, logger)    

    # - Planar Arm
    print("\n[*] Planar Arm on CVT w/Improvement Emitters")
    from bbq.examples.planar_arm import PlanarArm  
    p = load_config([arm_config, exp_config, cmame_config])
    logger = RibsLogger(p)
    domain = PlanarArm(**p)
    archive = map_elites(domain, p, logger)

    # - Planar Arm [Line]
    print("\n[*] Planar Arm on CVT w/Line Emitters")
    from bbq.examples.planar_arm import PlanarArm  
    p = load_config([arm_config, exp_config, line_config])
    logger = RibsLogger(p)
    domain = PlanarArm(**p)
    archive = map_elites(domain, p, logger)

    # - Planar Arm [Line and Improvement]
    print("\n[*] Planar Arm on CVT w/Line and Improvement Emitters")
    from bbq.examples.planar_arm import PlanarArm  
    p = load_config([arm_config, exp_config, mixed_config])
    p['n_workers'] = 2 # To test parallel
    logger = RibsLogger(p)
    domain = PlanarArm(**p)
    archive = map_elites(domain, p, logger)    
    