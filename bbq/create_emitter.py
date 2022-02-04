import numpy as np
from bbq.emitters.obj_emitter import ObjEmitter

# - Emitters ------------------------------------------------------------------#
def create_emitter(emitter_choice, archive, p):
    if (emitter_choice == ObjEmitter):
        emitters = [emitter_choice(
            archive,
            p,
            batch_size=p['n_batch'], # Individuals created per loop
            ) for _ in range(p['n_emitters'])]   
    else:
        emitters = [emitter_choice(
            archive,
            np.zeros(p['dof'])+0.5,
            p['iso_sigma'],
            bounds=[p['param_bounds']]*p['dof'],
            batch_size=p['n_batch'], # Individuals created per loop
            ) for _ in range(p['n_emitters'])]       
    return emitters     