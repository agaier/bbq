"""A simple test to make sure all the bbq plumbing works
"""

import fire
from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites
from bbq.utils import load_config
from bbq.examples.rastrigin import Rastrigin


def rast_test(*extra_config, rep=0):
    # Load Configuration
    base_config = 'config/d_rast.yaml'
    exp_config  = 'config/x_smoke.yaml'# ; exp_config  = '../config/x_test.yaml'   
    print(f"|*******|\nREP: {rep}\nCFG: {extra_config}\n|*******|") 
    config_list = [base_config] + [exp_config] + list(extra_config)
    p = load_config(config_list) # Convert yaml files into parameter dictionary

    logger = RibsLogger(p)                  # Logging and visualization
    domain = Rastrigin(**p)                 # Rastrigin Domain
    archive = map_elites(domain, p, logger) # Run MAP-Elites

    print("Done")

if __name__ == '__main__':
    fire.Fire(rast_test)