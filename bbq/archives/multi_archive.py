
import numpy as np
from bbq.archives._archive_base import RandomBuffer, readonly
from ribs.archives._elite import Elite


class MultiContainer:
    """A class for wrapping multiple containers"""
    def __init__(self, p, archive_type, seed=None):
        archive_p = [{**A, **{'n_dof': p['n_dof']}} for A in p['Archives']]
        self._archives = [archive_type(a_p) for a_p in archive_p]
        self.dtype = self._archives[0].dtype

        ## Randomness ##
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._rand_buf = None

    def initialize(self, solution_dim):
        self._rand_buf = RandomBuffer(self._seed)
        for archive in self._archives:
            archive.initialize(solution_dim)

    def add_batch(self, sols, objs, desc, metas):
        """Add batch of elites to all archives"""
        for archive in self._archives:
            archive.add_batch(sols, objs, desc, metas)

    def as_numpy(self, **kwargs):
        return [a.as_numpy(**kwargs) for a in self._archives]

    def get_random_elite(self):
        """Choose one random elite from all archives"""
        if self.empty:
            raise IndexError("No elements in any archive.")

        # Collect all available individuals as [archive_id,index] pairs
        indices = [archive._occupied_indices for archive in self._archives]
        n_occupied = [len(ind) for ind in indices]
        index_key = []
        for list_idx, n_occ in enumerate(n_occupied):
            list_id, index = list_idx*np.ones(n_occ), np.arange(n_occ)
            index_key += [np.vstack((list_id, index)).T]
        index_key = np.vstack(index_key).astype(int)

        # Choose one at random
        index = self._rand_buf.get(sum(n_occupied))
        arch_i = index_key[index,0]        
        elite_i = indices[arch_i][index_key[index,1]]
        archive = self._archives[arch_i]

        return Elite(
            readonly(archive._solutions[elite_i]),
                     archive._objective_values[elite_i],
            readonly(archive._behavior_values[elite_i]),
                     elite_i,
                     archive._metadata[elite_i],
        )

    @property
    def archives(self):
        return self._archives

    @property
    def num_elites(self):
        return [a.stats.num_elites for a in self._archives]

    @property
    def obj_mean(self):
        return [a.stats.obj_mean for a in self._archives]        

    @property
    def qd_score(self):
        return [a.stats.qd_score for a in self._archives]        
    
    @property
    def empty(self):
        """bool: Whether all archives are empty."""
        return all([archive.empty for archive in self._archives])


def get_archive_params(p):
    p_list = p['Archives']

    return p_list