#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Give adjacency matrix get from calculus of clusters or label in network
through the CA_Atoms tuple it give back the adjacency matrix representing the function for alignment
the main of this script take the list of residues and perform a kmean cluster 
and a louvain computing partition respective on dataframe and network

"""
import networkx as nx
import numpy as np
import pandas as pd

from ProtACon.modules.basics import (
    CA_Atom,
    get_AA_features_dataframe,
)
from ProtACon import process_contact
from ProtACon.modules.on_network.Collect_and_structure_data import (
    get_df_about_instability,
    get_list_of_edges,
    get_weight_for_edges,
    generate_index_df,
)
from ProtACon.modules.on_network.kmeans_computing_and_results import (
    dictionary_from_tuple,
    get_clusters_label,
)
from ProtACon.modules.on_network.networks_analysis import (
    add_louvain_community_attribute,
    add_weight_combination,
    get_the_complete_Graph,  # does not exist
    weight_on_edge,
)


__author__ = 'Renato Eliasy'
__email__ = 'renatoeliasy@gmail.com'


def process_kmeans_map(CA_Atoms: tuple[CA_Atom, ...]
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    from the list of residues it compute the kmeans on the dataframe containing these feature :
    it return both the map of solely kmeans_label communities, the labels on the contact map, and a dict of label
    Parameters:
    ----------
    CA_Atoms : tuple[CA_Atom, ...]
        the tuple of CA_Atom object
    Returns:
    -------
    kmeans_map: np.ndarray
        the binary map of the adjacency matrix
    louvain_map: np.ndarray
        the binary map of the adjacency matrix
    """

    AA_general_dataframe = get_AA_features_dataframe(CA_Atoms=CA_Atoms)
    aa_feature_DF = AA_general_dataframe.copy()

    node_labels = generate_index_df(CA_Atoms=CA_Atoms)
    AA_general_dataframe['AA_pos'] = node_labels
    AA_general_dataframe.set_index('AA_pos', inplace=True)

    columns_to_remove_x_Kmeans = [col_name for col_name in ['AA_Name', 'AA_Coords', 'AA_Rcharge_density',
                                                            'AA_Charge_Density', 'AA_web_groups'] if col_name in AA_general_dataframe.columns]
    AA_df_Kmeans = AA_general_dataframe.drop(
        columns=columns_to_remove_x_Kmeans)

    kmeans_tuple_labels, Kmeans_df = get_clusters_label(
        dataset=AA_df_Kmeans, cluster_feature=AA_df_Kmeans['AA_web_group'], scaler_option='std')
    kmeans_label_dict = dictionary_from_tuple(kmeans_tuple_labels)

    kmean_map_df = pd.DataFrame(
        0, index=Kmeans_df.index, columns=Kmeans_df.index)
    for i in kmean_map_df.index:
        for j in kmean_map_df.columns:
            if kmeans_label_dict[i] == kmeans_label_dict[j]:
                kmean_map_df.at[i, j] = 1
    kmean_map = kmean_map_df.to_numpy()

    *_, bin_contacts_map = process_contact.main(CA_Atoms=CA_Atoms)
    kmean_contact = kmean_map*bin_contacts_map

    return (kmean_map,
            kmean_contact,  # FIXME postpone to calculate with louvain map:
            kmeans_label_dict)
# add another function for the plots
    # get the proximity graph:
    # get the complete graph from nx
    # get the proximity as a subgraph
    #


def process_louvain_map(CA_Atoms: tuple[CA_Atom, ...],
                        weight_on_edge_attributes: tuple[float, float, float] = (
                            0, 1, 0),
                        resolution: float = 4.0
                        ) -> tuple[np.ndarray, np.ndarray, nx.Graph]:
    """
    from the list of residues it perform the calculus of louvain communities on the network,whose features are:
        - Nodes
        - Edges
    it also return the dict of communities
    Parameters:
    ----------
    CA_Atoms : tuple[CA_Atom, ...]
        the tuple of CA_Atom object
    weight_on_edge_attributes : tuple[float, float, float]
        the tuple of the weight on edge attributes from 
    Returns:
    -------
    louvain_map: np.ndarray
        the binary map of the adjacency matrix
    louvain_contact_map : dict
        the combination between louvain and contact map
    Graph_network : nx.Graph
        the networkx graph of the network
        #NOTE add feature of nodes and edges in the graph
    """
    # DATAFRAME X FEATURE OF NODE
    all_feature_dataframe = get_AA_features_dataframe(CA_Atoms=CA_Atoms)
    aa_feature_DF = all_feature_dataframe.copy()

    # change index for nodes labelling
    node_labels = generate_index_df(CA_Atoms=CA_Atoms)
    all_feature_dataframe['AA_pos'] = node_labels
    all_feature_dataframe.set_index('AA_pos', inplace=True)
    if 'AA_web_groups' in all_feature_dataframe.columns:
        tot_clusters = float(len(set(all_feature_dataframe['AA_web_groups'])))

    else:
        tot_clusters = 4

    columns_to_remove_for_louvain_communities = []

    # get the edges weights list:
    instability_df = get_df_about_instability(base=CA_Atoms)
    # NOTE to change the list of edges: consider bin_contact_map = np.ones(len(CA_Atoms), len(CA_Atoms))
    distances_map, _, bin_contact_map = process_contact.main(CA_Atoms=CA_Atoms)
    distancies_df = pd.DataFrame(
        distances_map, index=node_labels, columns=node_labels)
    list_of_edges, bin_contact_df = get_list_of_edges(
        base_map=bin_contact_map, CA_Atoms=CA_Atoms)    # NOTE to change the list of edges: consider bin_contact_map = np.ones(len(CA_Atoms), len(CA_Atoms))
    edges_and_respective_weights = get_weight_for_edges(
        list_of_edges=list_of_edges, base_df=distancies_df, instability_df=instability_df)

    network_Graph = get_the_complete_Graph(
        dataframe_of_features=all_feature_dataframe, edges_weight_list=edges_and_respective_weights)

    weight_of_edge_attributes = weight_on_edge(
        weight_on_edge_attributes[0], weight_on_edge_attributes[1], weight_on_edge_attributes[2])

    network_Graph_w_weight_combination = add_weight_combination(
        G=network_Graph, weight_on_edge=weight_of_edge_attributes)

    louvain_communities_Graph = add_louvain_community_attribute(G=network_Graph_w_weight_combination,
                                                                weight_of_edges='weight_combination',
                                                                resolution=(resolution/tot_clusters))  # it add the attribute louvain_community of nodes
    louvain_communities_Dict = nx.get_node_attributes(G=louvain_communities_Graph,
                                                      name='louvain_community')
    louvain_map_df = pd.DataFrame(
        0, index=louvain_communities_Dict.keys(), columns=louvain_communities_Dict.keys())
    for i in louvain_map_df.index:
        for j in louvain_map_df.columns:
            if louvain_communities_Dict[i] == louvain_communities_Dict[j] and i != j:
                louvain_map_df.at[i, j] = 1
    louvain_Map = louvain_map_df.to_numpy()
    louvain_Contact_Map = louvain_Map * bin_contact_map
    return (louvain_Map,
            louvain_Contact_Map,
            louvain_communities_Graph)


def main(CA_Atoms: tuple[CA_Atom, ...],
         weight_for_louvain_modularity: tuple[float, float, float] = (0, 1, 0)
         ) -> tuple:
    '''
    this function collect the result of the previous function to reassume the results
    Parameters:
    ----------
    CA_Atoms : tuple[CA_Atom, ...]
        the tuple of CA_Atom object
    weight_for_louvain_modularity : tuple[float, float, float]

    Returns:
    -------
    kmeans_map: np.ndarray
        the binary map of the adjacency matrix considering the kmeans cluster's labels
    louvain_map: np.ndarray
        the binary map of the adjacency matrix considering the louvain partitioning
    kmeans_contact_map: np.ndarray
        the binary map of the adjacency matrix considering the kmeans cluster's labels and the contact map
    louvain_contact_map: np.ndarray
        the binary map of the adjacency matrix considering the louvain partitioning and the contact map
    kmeans_label_dict: dict
        the dictionary of the kmeans labels to have an easy access to the clusters
    louvain_communities_Graph: nx.Graph
        the networkx graph of the network with the louvain communities and the kmeans cluster's label to be draw as resulted graph
    '''
    kmeans_map, kmeans_contact_map, kmeans_label_dict = process_kmeans_map(
        CA_Atoms=CA_Atoms)
    # resolutions:
    resolution = len(set(kmeans_label_dict.values()))
    louvain_map, louvain_contact_map, louvain_communities_Graph = process_louvain_map(
        CA_Atoms=CA_Atoms, weight_on_edge_attributes=weight_for_louvain_modularity, resolution=resolution)

    for node in louvain_communities_Graph.nodes:
        louvain_communities_Graph.nodes[node]['kmeans_label'] = kmeans_label_dict[node]

    return (kmeans_map,
            louvain_map,
            kmeans_contact_map,
            louvain_contact_map,
            kmeans_label_dict,
            louvain_communities_Graph)
