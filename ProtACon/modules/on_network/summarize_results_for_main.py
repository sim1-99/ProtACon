#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

'''
this script is useful to reassume the work on network, 
both for analysis and for visualization
its means is to slim the code to use in the __main__.py script
it works with
- the results of the network
- the result of kmeans
- the result of louvain
'''

from ProtACon.modules.on_network import networks_analysis, kmeans_computing_and_results, PCA_computing_and_results, Attention_map_from_networks, Collect_and_structure_data
from ProtACon.modules.miscellaneous import CA_Atom, get_AA_features_dataframe
from ProtACon.modules.contact import generate_distance_map, generate_instability_map
import pandas as pd
import networkx as nx
import numpy as np
# Define the results for kmeans


def get_kmeans_results(
        CA_Atoms: tuple[CA_Atom, ...],

) -> tuple[
    pd.DataFrame,  # the updated dataframe
    tuple[int, ...],  # kmeans_labels
]:
    '''
    It give the results of the kmeans analysis
    Parameters:
    ----------
    CA_Atoms: tuple[CA_Atom,...]
        the tuple of the CA_Atom objects

    Returns:
    -------
    tuple[pd.DataFrame, tuple[int,...]]
        the updated dataframe
        the kmeans_labels
    '''
    feature_df = get_AA_features_dataframe(CA_Atoms=CA_Atoms)
    kmeans_labels,  new_df = kmeans_computing_and_results.get_clusters_label(
        dataset=feature_df,
        cluster_feature=feature_df['AA_web_group']
    )
    return (new_df, kmeans_labels)

# summarize the steps to get the complete nx.Graph rapresentation of the protein


def prepare_complete_graph_nx(CA_Atoms: tuple[CA_Atom, ...],
                              binary_map: np.ndarray | None
                              ) -> nx.Graph:
    '''
    from the CA_Atoms list it's in need:
    - the AA_dataframe
    - the the instability value from DIWV dict
    - the distancies between AAs
    - the bool to see if they are in contact or not
    '''
    node_name_for_Graph = Collect_and_structure_data.generate_index_df(
        CA_Atoms=CA_Atoms)
    instability_df = Collect_and_structure_data.get_dataframe_from_nparray(base_map=generate_instability_map(
        CA_Atoms=CA_Atoms), index_str=node_name_for_Graph, columns_str=node_name_for_Graph)
    distance_df = Collect_and_structure_data.get_dataframe_from_nparray(base_map=generate_distance_map(
        CA_Atoms=CA_Atoms), index_str=node_name_for_Graph, columns_str=node_name_for_Graph)
    if binary_map == None:
        list_of_edges = []
        for i, AA_i in enumerate(node_name_for_Graph):
            for j, AA_j in enumerate(node_name_for_Graph):
                if i < j:
                    list_of_edges.append((AA_i, AA_j))
    else:
        list_of_edges, _ = Collect_and_structure_data.get_list_of_edges(
            base_map=binary_map, CA_Atoms=CA_Atoms)
        weights_for_edges = Collect_and_structure_data.get_weight_for_edges(
            list_of_edges=list_of_edges,
            base_df=distance_df,
            instability_df=instability_df)
    nx_graph = Collect_and_structure_data.get_the_Graph_network(
        CA_Atoms=CA_Atoms, edges_weight_list=weights_for_edges)
    return nx_graph


pass


# Define the results for louvain
def get_louvain_results():
    '''
    It give the results of the louvain analysis
    Returns:
    -------
    tuple[pd.DataFrame, tuple[int,...]]
        the updated dataframe
        the louvain_labels
    '''
    # final generate the dataframe with louvain communities
    # generate necessary object to use to create the base network
    pass
