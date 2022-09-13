"""A simple test to make sure all the bbq plumbing works
"""

import fire
from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites
from bbq.utils import load_config
from bbq.examples.rastrigin import Rastrigin


def rast_test(extra_config=None):
    # Load Configuration
    base_config = '../config/d_rast.yaml'
    exp_config  = '../config/x_smoke.yaml'
    #exp_config  = '../config/x_test.yaml'    
    config_list = [base_config, exp_config, extra_config]
    config_list = [i for i in config_list if i is not None]
    p = load_config(config_list) # Convert yaml files into parameter dictionary

    logger = RibsLogger(p)                  # Logging and visualization
    domain = Rastrigin(**p)                 # Rastrigin Domain
    archive = map_elites(domain, p, logger) # Run MAP-Elites

    print("Done")


if __name__ == '__main__':
    fire.Fire(rast_test)
    #fire.Fire(rast_test, 'config/line_cma_mix.yaml')