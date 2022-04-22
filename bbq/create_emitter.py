import numpy as np
from bbq.emitters.obj_emitter import ObjEmitter

# - Emitters ------------------------------------------------------------------#
def create_emitter(emitter_choice, archive, p):
    if emitter_choice == ObjEmitter:
        emitters = [emitter_choice(
            archive,
            p,
            batch_size=p['n_batch'], # Individuals created per loop
            ) for _ in range(p['n_emitters'])]   
    else:
        if type(p['iso_sigma']) == float:
            p['iso_sigma'] = np.ones(p['n_emitters'])*p['iso_sigma']

        emitters = [emitter_choice(
            archive,
            np.zeros(p['n_dof'])+0.5,
            p['iso_sigma'][i],
            bounds=[[0,1]]*p['n_dof'],
            batch_size=p['n_batch'], # Individuals created per loop
            ) for i in range(p['n_emitters'])]       
    return emitters     