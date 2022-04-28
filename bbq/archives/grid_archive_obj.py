from bbq.archives.grid_archive import GridArchive
from bbq.archives._archive_base_obj import ArchiveBase_Obj
import numpy as np

class GridArchive_Obj(GridArchive, ArchiveBase_Obj):
    def __init__(self, p, seed=None):
        super().__init__(p, seed)

    def add(self, solution, objective_value, behavior_values, metadata=None):
        super(GridArchive, self).add(solution, objective_value, behavior_values, metadata)

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
