from bbq.utils import create_config
from bbq.examples.arm import Arm
from bbq.examples.run_arm import run_me
import fire

CONFIG_PATH = '../../config/'


def launch_instance(id=2, rep=1):   
    base_config = CONFIG_PATH + 'arm.yaml'
    exp_config  = CONFIG_PATH + 'test.yaml'
    p = create_config(base_config, exp_config)

    if id == 1:    
        p['exp_name'] = 'Uniform Init - Rand Offset'
        domain = Arm(p, seed=0, slope=2)
        run_me(domain, p)
        
    elif id == 2:
        p['exp_name'] = 'Normal Init - Rand Offset'
        domain = Arm(p, seed=0, slope=2, uniform_init=False)
        run_me(domain, p)

    elif id == 3:
        p['exp_name'] = 'Uniform Init - No Offset'
        domain = Arm(p, seed=0, slope=1)
        run_me(domain, p)      

    elif id == 4:
        p['exp_name'] = 'Normal Init - No Offset'
        domain = Arm(p, seed=0, slope=1, uniform_init=False)
        run_me(domain, p)    

if __name__ == '__main__':
    fire.Fire(launch_instance)