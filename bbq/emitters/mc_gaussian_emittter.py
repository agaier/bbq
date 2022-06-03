from bbq.emitters._mc_emitter_base import MC_EmitterBase
from ribs.emitters._gaussian_emitter import GaussianEmitter
import itertools
import numpy as np

class GaussianEmitter(GaussianEmitter, MC_EmitterBase):
    def __init__(self, archives, x0, sigma0, bounds=None, batch_size=64, seed=None):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._x0 = np.array(x0)
        self._sigma0 = np.array(sigma0)

        MC_EmitterBase.__init__(
            self,
            archives,
            len(self._x0),
            bounds,
        )

    def ask(self):
        """Creates solutions by adding Gaussian noise to elites in the archive.

        If the archive is empty, solutions are drawn from a (diagonal) Gaussian
        distribution centered at ``self.x0``. Otherwise, each solution is drawn
        from a distribution centered at a randomly chosen elite. In either case,
        the standard deviation is ``self.sigma0``.

        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """
        # Get all possible parents


        # Select batch_size number

        if self.archive.empty:
            parents = np.expand_dims(self._x0, axis=0)
        else:
            parents = [
                self.archive.get_random_elite().sol
                for _ in range(self._batch_size)
            ]

        noise = self._rng.normal(
            scale=self._sigma0,
            size=(self._batch_size, self.solution_dim),
        ).astype(self.archive.dtype)

        return self._ask_clip_helper(np.asarray(parents), noise,
                                     self.lower_bounds, self.upper_bounds)