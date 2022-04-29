import numpy as np
from ribs.archives import GridArchive

class GridArchive(GridArchive):
    def __init__(self, p, seed=None, dtype=np.float64):
        self.p = p
        dims   = p['grid_res']
        ranges = p['desc_bounds']
        super().__init__(dims, ranges, seed=seed, dtype=dtype)

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

    def add_batch(self, xx, objs, descs, meta=None):
        if meta is not None:
            for i in range(len(objs)):
                self.add(xx[i], objs[i], descs[i], metadata=meta[i])
        else:
            for i in range(len(objs)):
                self.add(xx[i], objs[i], descs[i])    