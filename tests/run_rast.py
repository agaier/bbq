""" Simple rastrigin example using bbq pattern  """


from bbq.map_elites import map_elites
from bbq.logging.logger import RibsLogger
from ribs.emitters import GaussianEmitter

def run_me(domain, p):
    d = domain(p)
    logger = RibsLogger(p)
    archive = map_elites(d, p, logger, emitter_type=GaussianEmitter)    
    logger.zip_results()
    print('\n[*] Done')

if __name__ == '__main__':
    from bbq.utils import create_config
    from bbq.examples.domain_example import Rastrigin
    
    base_config = 'config/rast.yaml'
    exp_config = 'config/test.yaml'
    #exp_config = 'config/smoke.yaml'
    
    p = create_config(base_config, exp_config)

    run_me(Rastrigin, p)