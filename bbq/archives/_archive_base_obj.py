import numpy as np
from ribs.archives._archive_base import ArchiveBase, RandomBuffer, AddStatus, require_init

import numba as nb


class ArchiveBase_Obj(ArchiveBase):
    def __init__(self, storage_dims, behavior_dim, seed=None, dtype=np.float64):
        ArchiveBase.__init__(
            self,
            storage_dims=tuple(self._dims),
            behavior_dim=len(self._dims),
            seed=seed,
            dtype=dtype,
        )

    def initialize(self, solution_dim):
        """Initializes the archive by allocating storage space.

        Child classes should call this method in their implementation if they
        are overriding it.

        Args:
            solution_dim (int): The dimension of the solution space.
        Raises:
            RuntimeError: The archive is already initialized.
        """
        if self._initialized:
            raise RuntimeError("Cannot re-initialize an archive")
        self._initialized = True

        self._rand_buf = RandomBuffer(self._seed)
        self._solution_dim = solution_dim
        self._occupied = np.zeros(self._storage_dims, dtype=bool)
        self._solutions = np.empty((*self._storage_dims, solution_dim),
                                    dtype=object)
        self._objective_values = np.empty(self._storage_dims, dtype=self._dtype)
        self._behavior_values = np.empty(
            (*self._storage_dims, self._behavior_dim), dtype=self.dtype)
        self._metadata = np.empty(self._storage_dims, dtype=object)
        self._occupied_indices = []
        self._occupied_indices_cols = tuple(
            [] for _ in range(len(self._storage_dims)))

        self._stats_reset()
        self._state = {"clear": 0, "add": 0}

    @staticmethod
    @nb.jit(locals={"already_occupied": nb.types.b1}, nopython=True)
    def _add_numba(new_index, new_objective_value,
                   new_behavior_values, occupied, objective_values,
                   behavior_values):
        """Numba helper for inserting solutions into the archive.

        See add() for usage.

        Returns:
            was_inserted (bool): Whether the new values were inserted into the
                archive.
            already_occupied (bool): Whether the index was occupied prior
                to this call; i.e. this is True only if there was already an
                item at the index.
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

            return True, already_occupied

        return False, already_occupied            

    @require_init
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
        behavior_values = np.asarray(behavior_values)
        objective_value = self.dtype(objective_value)

        index = self.get_index(behavior_values)
        old_objective = self._objective_values[index]

        was_inserted, already_occupied = self._add_numba(
            index, objective_value, behavior_values, self._occupied,
            self._objective_values, self._behavior_values)

        if was_inserted:
            self._metadata[index] = metadata
            self._solutions[index] = solution

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
