""" Simple rastrigin example using bbq pattern  """

from bbq.logging.logger import RibsLogger
from bbq.map_elites import map_elites
from bbq.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.emitters import IsoLineEmitter
from ribs.emitters import ImprovementEmitter


def run_me(domain, p, emitter_type=GaussianEmitter, archive_type=GridArchive):
    d = domain(p)
    logger = RibsLogger(p)
    archive = map_elites(d, p, logger, emitter_type=emitter_type, 
                                       archive_type=archive_type)    
    print(d.offset)
    print('\n[*] Done')

if __name__ == '__main__':
    from bbq.utils import create_config
    from bbq.examples.arm import Arm

    
    base_config = 'config/arm.yaml'
    exp_config = 'config/test.yaml'
    exp_config = 'config/smoke.yaml'
    
    p = create_config(base_config, exp_config)

    run_me(Arm, p, ImprovementEmitter)
