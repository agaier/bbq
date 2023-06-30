import numpy as np
from bbq.emitters._standard_emitters import (Bbq_Cma, Bbq_Gauss, Bbq_Line,
                                             Bbq_Object)

emitter_lookup = {'Cma'    : Bbq_Cma, 
                  'Line'   : Bbq_Line,
                  'Gauss'  : Bbq_Gauss,
                  'Object' : Bbq_Object
                  }

def init_emitter(p, archive, start_xx=None, emitter_lookup=emitter_lookup):
    "Initializes emitters from list of emitter configs"
    emitters = []
    for emitter in p['emitters']:
        emitter['bounds'] = [p['param_bounds']]*p['n_params']
        emitter['x0'] = start_xx[np.random.randint(len(start_xx))]
        emitter_type = emitter_lookup[emitter['type']]
        emitters += [emitter_type(archive, **emitter)]
    return emitters
