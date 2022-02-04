"""Provides an Emitter for objects:

    - 'mutate' function must be supplied by class

"""
import numpy as np
from numba import jit

from ribs.emitters._emitter_base import EmitterBase



class ObjEmitter(EmitterBase):
    """Emits solutions by calling object class mutate function on 
         existing archive solutions.

    If the archive is empty, calls to :meth:`ask` will generate solutions from a
    user-specified Gaussian distribution with mean ``x0`` and standard deviation
    ``sigma0``. Otherwise, this emitter selects solutions from the archive and
    generates solutions from a Gaussian distribution centered around each
    solution with standard deviation ``sigma0``.

    This is the classic variation operator presented in `Mouret 2015
    <https://arxiv.org/pdf/1504.04909.pdf>`_.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty.
        sigma0 (float or array-like): Standard deviation of the Gaussian
            distribution, both when the archive is empty and afterwards. Note we
            assume the Gaussian is diagonal, so if this argument is an array, it
            must be 1D.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(self, archive, mutation_params, batch_size=64, seed=None):

        EmitterBase.__init__(
            self,
            archive,
            1,
            None,
            
        )
        self._p = mutation_params
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
            raise ValueError("Cannot ask on empty archive") #todo: init in obj
        else:
            parents = [
                self.archive.get_random_elite().sol[0]
                for _ in range(self._batch_size)
            ]

        children = []
        for parent in parents:
            children += [parent.mutate(self._p)]

        return children
