import numpy as np
from bbq.archives._grid_base import GridArchive

class GridArchive(GridArchive):
    def __init__(self, p, seed=None, is_object=False):
        self.p = p
        dims   = p['grid_res']
        ranges = p['desc_bounds']
        super().__init__(dims, ranges, seed=seed, is_object=is_object)

    def as_numpy(self, include_metadata=False):
        # Create array
        grid_res = [len(a)-1 for a in self.boundaries]
        n_channels = sum([1, self._behavior_dim, self.p['n_dof']])
        np_archive = np.full(np.r_[grid_res, n_channels], np.nan)

        # Fill array
        # --> TODO: work on higher dim grids
        for elite in self:
            elite_stats = np.r_[elite.obj, elite.beh, elite.sol]
            np_archive[elite.idx[0], elite.idx[1], :] = elite_stats
        if not include_metadata:
            return np_archive
 

class GridArchive_Obj(GridArchive):
    def __init__(self, p, seed=None):
        super().__init__(p, seed, is_object=True)

    def as_numpy(self, include_metadata=False):
        # Create array
        grid_res = [len(a)-1 for a in self.boundaries]
        n_channels = sum([1, self._behavior_dim, 1]) # objective, behavior, ind 
        np_archive = np.full(np.r_[grid_res, n_channels], np.nan, dtype=object)

        # Fill array
        for elite in self:
            elite_stats = np.r_[elite.obj, elite.beh, elite.sol]
            np_archive[elite.idx[0], elite.idx[1], :] = elite_stats
        if not include_metadata:
            return np_archive        
