""" Simple rastrigin example using bbq pattern  """

from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites
from bbq.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.emitters import IsoLineEmitter
from ribs.emitters import ImprovementEmitter

def run_me(domain, p, archive_type=GridArchive):
    logger = RibsLogger(p, clear=False)
    archive = map_elites(domain, p, logger, archive_type=archive_type)    
    print(domain.offset)
    print('\n[*] Done')

if __name__ == '__main__':
    from bbq.utils import create_config
    from bbq.examples.arm import Arm
    CONFIG_PATH = '../../config/'
    
    base_config = CONFIG_PATH + 'arm.yaml'
    exp_config  = CONFIG_PATH + 'test.yaml'
    #exp_config  = CONFIG_PATH + 'smoke.yaml'
    
    p = create_config(base_config, exp_config)

    p['exp_name'] = 'Uniform Init - Rand Offset'
    domain = Arm(p, seed=0, slope=2)
    run_me(domain, p)

    p['exp_name'] = 'Normal Init - Rand Offset'
    domain = Arm(p, seed=0, slope=2, uniform_init=False)
    run_me(domain, p)

    p['exp_name'] = 'Uniform Init - No Offset'
    domain = Arm(p, seed=0, slope=1)
    run_me(domain, p)

    p['exp_name'] = 'Normal Init - No Offset'
    domain = Arm(p, seed=0, slope=1, uniform_init=False)
    run_me(domain, p)    

