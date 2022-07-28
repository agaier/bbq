"""Modifications of ArchiveBase for BBQ Features."""

import numpy as np
from ribs.archives._add_status import AddStatus
from ribs.archives._archive_base import ArchiveBase


class BBQArchiveBase(ArchiveBase):
    def __init__(self, storage_dims, behavior_dim, seed=None, dtype=np.float64):
        super().__init__(storage_dims, behavior_dim, seed, dtype)

    def initialize(self, solution_dim):
        if self.use_objects is True:
            self._sol_dtype = object
            super().initialize(1)
        else:
            self._sol_dtype = self.dtype
            super().initialize(solution_dim)            
        self._solutions = np.empty((*self._storage_dims, solution_dim),
                            dtype=self._sol_dtype)
        self._objective_values = np.full((self._storage_dims), np.nan)                            

    def add_batch(self, xx, objs, descs, meta=None):
        """ Add set of solutions at once"""
        if meta is not None:
            for i in range(len(objs)):
                self.add(xx[i], objs[i], descs[i], metadata=meta[i])
        else:
            for i in range(len(objs)):
                self.add(xx[i], objs[i], descs[i])   

    #@require_init
    def add(self, solution, objective_value, behavior_values, metadata=None):
        """Attempts to insert a new solution into the archive.

        The solution is only inserted if it has a higher ``objective_value``
        than the elite previously in the corresponding bin.

        Args:
            solution (array-like): Parameters of the solution.
            objective_value (float): Objective function evaluation of the
                solution.
            behavior_values (array-like): Coordinates in behavior space of the
                solution.
            metadata (object): Any Python object representing metadata for the
                solution. For instance, this could be a dict with several
                properties.
        Returns:
            tuple: 2-element tuple describing the result of the add operation.
            These outputs are particularly useful for algorithms such as CMA-ME.

                **status** (:class:`AddStatus`): See :class:`AddStatus`.

                **value** (:attr:`dtype`): The meaning of this value depends on
                the value of ``status``:

                - ``NOT_ADDED`` -> the "negative improvement," i.e. objective
                  value of solution passed in minus objective value of the
                  solution still in the archive (this value is negative because
                  the solution did not have a high enough objective value to be
                  added to the archive)
                - ``IMPROVE_EXISTING`` -> the "improvement," i.e. objective
                  value of solution passed in minus objective value of solution
                  previously in the archive
                - ``NEW`` -> the objective value passed in
        """
        self._state["add"] += 1
        solution = np.asarray(solution)
        behavior_values = np.asarray(behavior_values)
        objective_value = self.dtype(objective_value)

        index = self.get_index(behavior_values)
        old_objective = self._objective_values[index]
        was_inserted, already_occupied = self.insert(
            index, solution, objective_value, behavior_values, metadata)

        if was_inserted and not already_occupied:
            self._add_occupied_index(index)
            status = AddStatus.NEW
            value = objective_value
            self._stats_update(self.dtype(0.0), objective_value)
        elif was_inserted and already_occupied:
            status = AddStatus.IMPROVE_EXISTING
            value = objective_value - old_objective
            self._stats_update(old_objective, objective_value)
        else:
            status = AddStatus.NOT_ADDED
            value = objective_value - old_objective
        return status, value

    def insert(self, index, solution, objective_value, behavior_values, metadata):
        """ Cut out of main add function to allow variations, e.g. T-DominO """
        was_inserted, already_occupied = self._add_to_bin(
            index, solution, objective_value, behavior_values, self._occupied,
            self._solutions, self._objective_values, self._behavior_values)

        if was_inserted:
            self._metadata[index] = metadata

        return was_inserted, already_occupied

    def _add_to_bin(self, new_index, new_solution, new_objective_value,
                   new_behavior_values, occupied, solutions, objective_values,
                   behavior_values):
        """Helper for inserting solutions into the archive.

        See add() for usage.

        Returns:
            was_inserted (bool): Whether the new values were inserted into the
                archive.
            already_occupied (bool): Whether the index was occupied prior
                to this call; i.e. this is True only if there was already an
                item at the index.

        BBQ: Assignment is from a bottleneck -- numba just blocks flexibility
             by not allowing objects.
        """
        already_occupied = occupied[new_index]
        if (not already_occupied or
                objective_values[new_index] < new_objective_value):
            # Track this index if it has not been seen before -- important that
            # we do this before inserting the solution.
            if not already_occupied:
                occupied[new_index] = True

            # Insert into the archive.
            objective_values[new_index] = new_objective_value
            behavior_values[new_index] = new_behavior_values
            solutions[new_index] = new_solution

            return True, already_occupied

        return False, already_occupied