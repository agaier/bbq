"""Provides an Emitter for objects:

    - 'mutate' function must be supplied by class

"""
import numpy as np
from bbq.emitters._emitter_base import Bbq_EmitterBase

class ObjEmitter(Bbq_EmitterBase):
    """Emits solutions by calling object class mutate function on 
         existing archive solutions.
    """

    def __init__(self, archive, mut_p, batch_size=64, seed=None, name='--', **_):
        Bbq_EmitterBase.__init__(self, archive, 1, None)
        self.name = name
        self._p = mut_p
        self._batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        

    @property
    def p(self):
        """Mutation parameters"""
        return self._p

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def ask(self):
        """Creates solutions by calling object class mutate function 

        Returns:
            ``(batch_size)`` array -- contains ``batch_size`` new solutions to 
                                      evaluate.
        """
        if self.archive.empty:
            raise ValueError("Cannot ask on empty archive")
        else:
            parents = [
                self.archive.get_random_elite().sol[0]
                for _ in range(self._batch_size)
            ]
        children = [parent.mutate(self._p) for parent in parents]
        return children