from bbq.archives.grid_archive import GridArchive
from bbq.archives._archive_base_obj import ArchiveBase_Obj

class GridArchive_Obj(GridArchive, ArchiveBase_Obj):
    def __init__(self, p, seed=None):
        super().__init__(p, seed)

    def add(self, solution, objective_value, behavior_values, metadata=None):
        super(GridArchive, self).add(solution, objective_value, behavior_values, metadata)