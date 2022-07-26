import numpy as np
from bbq.emitters._standard_emitters import ImprovementEmitter, IsoLineEmitter, GaussianEmitter

emitter_lookup = {'Improvement' : ImprovementEmitter, 
                  'Line'        : IsoLineEmitter,
                  'Gaussian'    : GaussianEmitter}


def init_emitters(p, archive, start_xx=None):
    "Intializes emitters from list of emitter configs"
    emitters = []
    for emitter in p['emitters']:
        emitter['bounds'] = [p['param_bounds']]*p['n_dof']
        emitter['x0'] = start_xx[np.random.randint(start_xx.shape[0])]
        emitter_type = emitter_lookup[emitter['type']]
        emitters += [emitter_type(archive, **emitter)]
    return emitters