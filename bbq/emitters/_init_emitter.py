import numpy as np
from bbq.emitters._standard_emitters import (GaussianEmitter,
                                             ImprovementEmitter,
                                             IsoLineEmitter)
from bbq.emitters.obj_emitter import ObjEmitter

emitter_lookup = {'Improvement' : ImprovementEmitter, 
                  'Line'        : IsoLineEmitter,
                  'Gaussian'    : GaussianEmitter,
                  'Object'      : ObjEmitter
                  }

def init_emitter(p, archive, start_xx=None):
    "Intializes emitters from list of emitter configs"
    emitters = []
    for emitter in p['emitters']:
        emitter['bounds'] = [p['param_bounds']]*p['n_dof']
        emitter['x0'] = start_xx[np.random.randint(len(start_xx))]
        emitter_type = emitter_lookup[emitter['type']]
        emitters += [emitter_type(archive, **emitter)]
    return emitters
