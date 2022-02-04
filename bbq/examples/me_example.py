"""Example usage of using PyRibs framework with helper rools in ribs_helpers
"""
import fire
import yaml


from bbq.map_elites import map_elites
from bbq.logging.logger import RibsLogger
from domain_example import Rastrigin

from ribs.emitters import IsoLineEmitter, ImprovementEmitter

def run_me(config_file='rast_config.yaml'):
    p = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    domain = Rastrigin(n_dof=p['dof'], 
                       n_desc=len(p['desc_bounds']),
                       x_scale=p['param_bounds'])
    logger = RibsLogger(p, save_meta=True, copy_config=config_file, clear=False)
    archive = map_elites(domain, p, logger, emitter_type=IsoLineEmitter)
    
    logger.zip_results()
    print('\n[*] Done')

if __name__ == '__main__':
    fire.Fire(run_me)
