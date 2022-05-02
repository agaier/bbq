import numpy as np
from ribs.archives._archive_base import RandomBuffer, AddStatus, require_init
from bbq.archives._archive_base import ArchiveBase

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

    def insert(self, index, solution, objective_value, behavior_values, metadata):
        """ BBQ: Variation which inserts solution without numba """
        was_inserted, already_occupied = self._add_numba(
            index, objective_value, behavior_values, self._occupied,
            self._objective_values, self._behavior_values)

        if was_inserted:
            self._metadata[index] = metadata
            self._solutions[index] = solution

        return was_inserted, already_occupied

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