#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def binary_map_from_kmeans(proximity_graph: nx.Graph,
                           kmean_label_dict: dict,
                           ) -> np.ndarray:
    """
    this function get the edge of the communities individuated by kmean clusters
    and traspose them into an adjacency matrix
    Parameters:
    ----------
    proximity_graph: nx.Graph
        the graph containing only the edges['lenght'] < 7Angstrom
    kmean_label_dict: dict
        the dictionary containing the label of the clusters obtained through the dictionary_from_tuple function in kmeans module

    Returns:
    -------
    binary_map: np.ndarray
        the binary map of the adjacency matrix
    """

    edge_community_list = []
    for i, j in proximity_graph.edges(data=True):
        if kmean_label_dict[i] == kmean_label_dict[j]:
            edge_community_list.append((i, j))

    community_map = pd.DataFrame(
        0, index=proximity_graph.nodes, columns=proximity_graph.nodes)
    for i, j in edge_community_list:
        community_map.at[i, j] = 1

    community_attention_map = community_map.to_numpy()
    return community_attention_map
