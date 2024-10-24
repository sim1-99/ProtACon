#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ProtACon.modules.on_network import networks_analysis as netly


def binary_map_from_clusters(proximity_graph: nx.Graph,
                             edge_feature: str = '',
                             node_feature: str = 'louvain_community',
                             same: bool = True,
                             ) -> np.ndarray:
    """
    this function get the edge of the communities individuated by clusters and community methods
    and traspose them into an adjacency matrix

    Parameters
    ----------
    proximity_graph: nx.Graph
        the graph containing only the edges['lenght'] < 7Angstrom
    edge_feature: str
        the feature to get from the graph in order to obtain the binary map
    node_feature: str
        the feature to get from the graph in order to obtain the binary map
    same: bool
        if True the function will return the binary map of the same community, otherwise the binary map of the different community

    Returns
    -----------
    binary_map: np.ndarray
        the binary map of the adjacency matrix
    """
    if edge_feature != '':
        edge_attribute_list, noproblem = netly.get_edge_attribute_list(
            G=proximity_graph, attribute_to_be_in=edge_feature)
        if not noproblem:
            raise ValueError('the attribute is not in the graph edge list')
    node_attributes, is_in = netly.get_node_atttribute_list(
        G=proximity_graph, attribute_to_be_in=node_feature)
    if not is_in:
        raise ValueError(
            f'the attribute {node_feature} is not in the graph node attribute list')
    louvain_community_dict = nx.get_node_attributes(
        G=proximity_graph, name=node_feature)

    community_map = pd.DataFrame(
        0, index=proximity_graph.nodes, columns=proximity_graph.nodes)
    for s in louvain_community_dict.keys():
        for t in louvain_community_dict.keys():
            if s == t:
                continue
            if louvain_community_dict[s] == louvain_community_dict[t] and same:
                community_map.at[s, t] = 1
            elif louvain_community_dict[s] != louvain_community_dict[t] and not same:
                community_map.at[s, t] = 1

    community_attention_map = community_map.to_numpy()
    return community_attention_map
