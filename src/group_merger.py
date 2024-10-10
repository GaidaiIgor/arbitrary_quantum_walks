import copy
import itertools as it

import exact_cover as ec
import numpy as np
from numpy import ndarray

from src.edge_group import EdgeGroup


class GroupMerger:
    """ Class for merging groups of edges into bigger groups that can be done in one gate. """

    @staticmethod
    def find_all_groups(edge_groups: list[EdgeGroup]) -> list[EdgeGroup]:
        """
        Finds all possible groups of n-dimensional edges that could be implemented in one gate.
        :param edge_groups: List of edge groups of a quantum walk on a hypercube that can be implemented in one gate (to be modified).
        :return: List of edge groups where some of the groups may be merged if they can be simultaneously implemented in one gate.
        """
        all_groups = edge_groups
        last_group = all_groups
        next_group = []
        while len(last_group) > 1:
            for pair in it.combinations(last_group, 2):
                different_dimensions = pair[0].coordinates != pair[1].coordinates
                if sum(different_dimensions) == 1:
                    group_copy = copy.deepcopy(pair[0])
                    dim = np.where(different_dimensions)[0][0]
                    group_copy.coordinates[dim] = -1
                    group_copy.edge_inds.extend(pair[1].edge_inds)
                    next_group.append(group_copy)
            all_groups.extend(next_group)
            last_group = next_group
            next_group = []
        return all_groups

    @staticmethod
    def convert_groups_to_cover_matrix(groups: list[EdgeGroup], total_edges: int) -> ndarray:
        """
        Converts edge groups into cover matrix, i.e. boolean matrix where [i, j] elem is true if set i covers item j and false otherwise.
        :param groups: List of edge groups.
        :param total_edges: Total number of edge_groups used to form given edge groups.
        :return: Cover matrix.
        """
        cover_matrix = np.zeros((len(groups), total_edges), dtype=bool)
        for i, group in enumerate(groups):
            cover_matrix[i, group.edge_inds] = True
        return cover_matrix

    def merge_groups(self, edge_groups: list[EdgeGroup]) -> list[EdgeGroup]:
        """
        Merges existing edge groups into bigger groups that can still be implemented in 1 gate.
        :var edge_groups: Existing edge groups (to be modified; typically single edges).
        :return: Merged groups.
        """
        all_groups = self.find_all_groups(edge_groups)
        cover_matrix = self.convert_groups_to_cover_matrix(all_groups, len(edge_groups))
        cover_group_inds = ec.get_exact_cover(cover_matrix)
        return [all_groups[i] for i in cover_group_inds]
