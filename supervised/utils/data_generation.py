# this file is to generate different distributed TSP/CVRP instances with different size including
# size: arbitrary specified
# distribution:
#     uniform: each node is uniformly i.i. distributed
#     clustered: uniformly distributed center with Gaussian distributed nodes around each center
#     explosion: uniformly distributed nodes, but not in
#     implosion: uniformly distributed nodes, but not in
# for every output, it is processed by min-max normalization over a [0,1]^2 unit board
# the output instances are given in torch.Tensor
#
# IMPLEMENTATION NOTES:
#     due to some reasons, instance generation are performed in something like for-loops
#     but this should not impose any inconvenience, since data generation it is not time-consuming

import sys

import torch
from torch.distributions import Exponential

sys.path.append(f"utils/")


def normalize_tsp_to_unit_board(tsp_instance):
    """
    normalize a tsp instance to a [0, 1]^2 unit board, prefer to have points on both x=0 and y=0
    :param tsp_instance: a (tsp_size, 2) tensor
    :return: a (tsp_size, 2) tensor, a normalized tsp instance
    """
    normalized_instance = tsp_instance.clone()
    normalization_factor = (normalized_instance.max(dim=0).values - normalized_instance.min(dim=0).values).max()
    normalized_instance = (normalized_instance - normalized_instance.min(dim=0).values) / normalization_factor
    return normalized_instance


def generate_uniform_tsp_instance(tsp_size):
    """
    uniformly i.i.d. coordinates of data points
    :param tsp_size: the number of nodes for tsp instances
    :return: a (tsp_size, 2) tensor, a tsp instance following uniform distribution
    """
    return normalize_tsp_to_unit_board(torch.rand((tsp_size, 2)))


def generate_clustered_tsp_instance(tsp_size, cluster_num=3, cluster_diversity=10):
    """
    uniformly i.i.d. coordinates of virtual cluster centers
    Gaussian coordinates of data points w.r.t. virtual cluster centers
    NOTE: set cluster_num=1 gives the normal distributed tsp instance
    :param tsp_size: the number of nodes for tsp instances
    :param cluster_num: number of clusters (generated virtual cluster center)
    :param cluster_diversity: the bound for generating clusters
    :return: a (tsp_size, 2) tensor, a tsp instance following clustered distribution
    """
    centers = cluster_diversity * torch.rand((cluster_num, 2))
    nodes = torch.normal(0, 1, (tsp_size, 2))
    group = torch.randint(cluster_num, (tsp_size,))
    tsp_instance = nodes + centers[group]
    return normalize_tsp_to_unit_board(tsp_instance)


def generate_explosion_tsp_instance(tsp_size, range_min=0.1, range_max=0.5, rate=10):
    """
    first generate uniformly i.i.d. coordinates of data points
    select an explosion center and expel all data points in a range
    :param tsp_size: the number of nodes for tsp instances
    :param range_min: the minimum range of explosion
    :param range_max: the maximum range of explosion
    :param rate: rate of exponential distribution, for random extra movement our of range
    :return: a (tsp_size, 2) tensor, a tsp instance following explosion distribution
    """
    tsp_instance = torch.rand((tsp_size, 2))
    explosion_center = torch.rand(2, )
    pointer_vector = tsp_instance - explosion_center
    explosion_range = (range_max - range_min) * torch.rand((1,)) + range_min
    exploded = pointer_vector.norm(dim=1) < explosion_range
    explosion_factor = explosion_range + Exponential(rate=rate).sample((tsp_size, 1))
    directional_vector = pointer_vector / pointer_vector.norm(dim=1).unsqueeze(dim=1)
    explosion_movement = directional_vector * explosion_factor
    tsp_instance[exploded] = explosion_center + explosion_movement[exploded]
    return normalize_tsp_to_unit_board(tsp_instance)


def generate_implosion_tsp_instance(tsp_size, range_min=0.1, range_max=0.5):
    """
    first generate uniformly i.i.d. coordinates of data points
    select an implosion center and attracts all data points in a range
    :param tsp_size: the number of nodes for tsp instances
    :param range_min: the minimum range of implosion
    :param range_max: the maximum range of implosion
    :return: a (tsp_size, 2) tensor, a tsp instance following implosion distribution
    """
    tsp_instance = torch.rand((tsp_size, 2))
    implosion_center = torch.rand(2, )
    pointer_vector = tsp_instance - implosion_center
    implosion_range = (range_max - range_min) * torch.rand((1,)) + range_min
    imploded = pointer_vector.norm(dim=1) < implosion_range
    implosion_factor = min(implosion_range, torch.normal(0, 1, (1,)))
    implosion_movement = pointer_vector * implosion_factor
    tsp_instance[imploded] = implosion_center + implosion_movement[imploded]
    return normalize_tsp_to_unit_board(tsp_instance)
