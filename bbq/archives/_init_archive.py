""" BBQ Archive versions. """


from bbq.archives._archive_base import BBQArchiveBase
from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._grid_archive import GridArchive

class BbqGrid(GridArchive, BBQArchiveBase):
    def __init__(self, grid_res=None, desc_bounds=None, seed=None, use_objects=False, **_):    
        self.use_objects = use_objects
        super().__init__(grid_res, desc_bounds)

class BbqCVT(CVTArchive, BBQArchiveBase):
    def __init__(self, n_bins=None, desc_bounds=None, seed=None, use_objects=False, **_):    
        self.use_objects = use_objects
        super().__init__(n_bins, desc_bounds)


archive_lookup = {'Grid'    : BbqGrid,
                  'CVT'     : BbqCVT}

def init_archive(p):
    """ Initializes archives from archive yaml config"""
    a_config = p['archive']
    archive_type = archive_lookup[a_config['type']]
    archive = archive_type(**a_config)
    return archive


