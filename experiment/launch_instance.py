from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites
from bbq.examples.planar_arm import PlanarArm  
from bbq.utils import load_config
import fire

def launch_instance(id=2, rep=1):
    # Experiment Setup
    config_dir = '../config/'
    base = 'd_arm_grid.yaml'
    exp = 'x_smoke.yaml' # test pipeline
    exp = 'x_test.yaml'  # full experiment
    
    exp_name = ['Gaussian', 'Line', 'CMA-ME', 'CMA+Line']
    emmiter = ['e_gauss.yaml', 'e_line.yaml', 'e_cmame.yaml', 'e_mixed.yaml']

    # Run Experiments
    p = load_config([config_dir+base, config_dir+exp, config_dir+emmiter[id]])
    p['exp_name'] = exp_name[id]
    logger = RibsLogger(p, rep=rep)
    domain = PlanarArm(**p)
    archive = map_elites(domain, p, logger)

if __name__ == '__main__':
    fire.Fire(launch_instance)