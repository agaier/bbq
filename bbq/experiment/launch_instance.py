from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites
from bbq.examples.planar_arm import PlanarArm  
from bbq.utils import load_config
import fire

def launch_instance(id=2, rep=1):
    # Experiment Setup
    config_dir = 'config/'
    base = 'arm.yaml'
    exp_name = ['Gaussian', 'Line', 'CMA-ME', 'CMA+Line']
    exp = ['gaus.yaml', 'line.yaml', 'cmame.yaml', 'cma_line.yaml']

    # Run Experiments
    p = load_config([config_dir+base, config_dir+exp[id]])
    p['exp_name'] = exp_name[id]
    logger = RibsLogger(p, rep=rep)
    domain = PlanarArm(**p)
    archive = map_elites(domain, p, logger)

if __name__ == '__main__':
    fire.Fire(launch_instance)