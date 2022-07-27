"""Gaussian Emitter from PyRibs with BBQ logging"""
from bbq.emitters._emitter_base import BBQ_EmitterBase
from ribs.emitters._gaussian_emitter import GaussianEmitter
from ribs.emitters._iso_line_emitter import IsoLineEmitter
from ribs.emitters._improvement_emitter import ImprovementEmitter

import itertools
from ribs.archives import AddStatus
import numpy as np


class GaussianEmitter(GaussianEmitter, BBQ_EmitterBase):
    def __init__(self, archive, x0, sigma0, bounds=None, batch_size=64, seed=None, name='--', **_):
        super().__init__(archive, x0, sigma0, bounds, batch_size, seed)
        self.name = name


class IsoLineEmitter(IsoLineEmitter, BBQ_EmitterBase):
    def __init__(self, archive, x0, iso_sigma=0.01, line_sigma=0.2, bounds=None, batch_size=64, seed=None, name='--', **_):
        super().__init__(archive, x0, iso_sigma, line_sigma, bounds, batch_size, seed)
        self.name = name


class ImprovementEmitter(ImprovementEmitter, BBQ_EmitterBase):
    def __init__(self, archive, x0, sigma0, selection_rule="filter", restart_rule="no_improvement", weight_rule="truncation", bounds=None, batch_size=None, seed=None, name='--', **_):
        super().__init__(archive, x0, sigma0, selection_rule, restart_rule, weight_rule, bounds, batch_size, seed)
        self.name = name


    def tell(self, solutions, objective_values, behavior_values, metadata=None):
        """Gives the emitter results from evaluating solutions.

    As solutions are inserted into the archive, we record their "improvement
    value" -- conveniently, this is the ``value`` returned by
    :meth:`ribs.archives.ArchiveBase.add`. We then rank the solutions
    according to their add status (new solutions rank in front of
    solutions that improved existing entries in the archive, which rank
    ahead of solutions that were not added), followed by their improvement
    value.  We then pass the ranked solutions to the underlying CMA-ES
    optimizer to update the search parameters.

    Args:
        solutions (numpy.ndarray): Array of solutions generated by this
            emitter's :meth:`ask()` method.
        objective_values (numpy.ndarray): 1D array containing the objective
            function value of each solution.
        behavior_values (numpy.ndarray): ``(n, <behavior space dimension>)``
            array with the behavior space coordinates of each solution.
        metadata (numpy.ndarray): 1D object array containing a metadata
            object for each solution.
    """
        pulse = np.zeros(3) # NOT_ADDED | IMPROVE | NEW
        ranking_data = []
        new_sols = 0
        metadata = itertools.repeat(None) if metadata is None else metadata
        for i, (sol, obj, beh, meta) in enumerate(
                zip(solutions, objective_values, behavior_values, metadata)):
            status, value = self.archive.add(sol, obj, beh, meta)
            ranking_data.append((status, value, i))
            if status in (AddStatus.NEW, AddStatus.IMPROVE_EXISTING):
                new_sols += 1
            pulse[status] += 1
        self.pulse = np.vstack([self.pulse, pulse])
        
        # New solutions sort ahead of improved ones, which sort ahead of ones
        # that were not added.
        ranking_data.sort(reverse=True)
        indices = [d[2] for d in ranking_data]

        num_parents = (new_sols if self._selection_rule == "filter" else
                    self._num_parents)

        self.opt.tell(solutions[indices], num_parents)

        # Check for reset.
        if (self.opt.check_stop([value for status, value, i in ranking_data]) or
                self._check_restart(new_sols)):
            new_x0 = self.archive.get_random_elite().sol
            self.opt.reset(new_x0)
            self._restarts += 1
                    

        

