#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import numpy as np
import pandas as pd
import networkx as nx
import logging
from ProtACon.modules.on_network.Collect_and_structure_data import generate_index_df
from sklearn.preprocessing import MinMaxScaler
from ProtACon.modules.miscellaneous import CA_Atom
from sklearn.metrics import homogeneity_completeness_v_measure
from typing import List, Tuple
'''
this script analyze the amminoacids in the protein, it also enhance some selected features
through colors
the dataframe used here has these basics columns:
       'AA_Name', 'AA_KD_hydrophobicity', 'AA_Volume', 'AA_Charge', 'AA_isoPH',
       'AA_Charge_Density', 'AA_Rcharge_density', 'AA_Xcoords', 'AA_Ycoords',
       'AA_Zcoords', 'AA_self_Flexibility', 'AA_HW_hydrophylicity',
       'AA_JA_in->out_E.transfer', 'AA_EM_surface.accessibility',
       'AA_local_flexibility', 'aromaticity', 'web_groups',
       'secondary_structure', 'vitality', 'AA_pos'
'''


def rescale_0_to_1(array: list | tuple | np.ndarray
                   ) -> list:
    """
    a function to rescale the values of an array from 0 to 1
    Parameters:
    -----------
    array : list | tuple | np.ndarray
        the array to be rescaled

    Returns:
    --------
    rescaled_array : list
        the rescaled array
    """
    scaler = MinMaxScaler()
    rescaled_array = scaler.fit_transform(np.array(array).reshape(-1, 1))
    return (rescaled_array)


def collect_results_about_partitions(homogeneity: float,
                                     completeness: float
                                     ) -> tuple[float, ...]:
    '''
    starting from homogeneity and completeness it returns a triplet tuple to collect
    and compute calculation about the V-measure:

    Parameters:
    ----------
    homogeneity: float
        the homogeneity score
    completeness: float
        the completeness score

    Returns:
    -------
    partitions_results : tuple[float,...]
        the V-measure, the homogeneity and the completeness
    '''
    V_measure = (homogeneity*completeness)/np.sum(homogeneity, completeness)
    V_measure = np.round(V_measure, 2)
    partitions_results = (V_measure, homogeneity, completeness)
    return partitions_results


def confront_partitions(partition_to_confront: dict | list | tuple,
                        # the web groups
                        CA_Atoms: tuple[CA_Atom, ...],
                        ground_truth: dict | tuple | list = None,
                        ) -> tuple[float, float, float]:
    """
    this calculate homogeneity and completness respecting the ground truth of web_group 
    and the partition to confront
    Parameters:
    ----------
    ground_truth : list[list[str],list[str], list[str], list[str]] | dict
        the web_groups= {
    'A' : 'G1', 'L' : 'G1', 'I' : 'G1', 'V' : 'G1', 'P' : 'G1', 'M' : 'G1', 'F' : 'G1', 'W' : 'G1',
    'S' : 'G2', 'T' : 'G2', 'Y' : 'G2', 'N' : 'G2', 'Q' : 'G2', 'C' : 'G2', 'G' : 'G2',
    'K' : 'G3', 'H' : 'G3', 'R' : 'G3', 
    'D' : 'G4', 'E' : 'G4'
    } as default, otherwise it has to have the keys with C(0) formats

    partition_to_confront : dict
        the partition to confront coming from kmeans or louvain's communities

    Returns:
    -------
    homogeneity : float
        the homogeneity score
    completeness : float
        the completeness score
    V_measure : float
        the V-measure score
    """
    web_groups = {
        'A': 1, 'L': 1, 'I': 1, 'V': 1, 'P': 1, 'M': 1, 'F': 1, 'W': 1,
        'S': 2, 'T': 2, 'Y': 2, 'N': 2, 'Q': 2, 'C': 2, 'G': 2,
        'K': 3, 'H': 3, 'R': 3,
        'D': 4, 'E': 4
    }
    index_label = generate_index_df(CA_Atoms=CA_Atoms)
    ground = []
    if ground_truth == None:
        ground_truth = []
        for index in index_label:
            ground.append(web_groups[index[0].upper()])
    else:
        ground = tuple(ground_truth.values())
    expected_partition = []
    if list(partition_to_confront.keys())[0] in web_groups.keys():
        for aa in partition_to_confront.keys():
            expected_partition.append(web_groups[aa])
    else:
        if isinstance(partition_to_confront, dict):
            expected_partition = tuple(partition_to_confront.values())
    homogeneity, completeness, V_measure = homogeneity_completeness_v_measure(labels_pred=expected_partition,
                                                                              labels_true=ground)
    return homogeneity, completeness, V_measure


# for confront partition, get the dataframe of protein.index to get the order of labels.


def compute_proximity_Graph(base_Graph: nx.Graph,
                            cut_off_distance: float,  # use the cut off of config.txt as default
                            feature: str = 'lenght',
                            threshold: float | str = 'zero',
                            ) -> nx.Graph:
    '''
    this function filter the edge in the complete graph: base_Graph
    to get a graph without the edge outside the threshold
    Parameters:
    ----------
    base_Graph: nx.Graph
        the complete graph to be filtered
        #FIXME add feature of edges and nodes
    cut_off_distance: float
        the distance to be used as threshold
    threshold : str
        the type of threshold to be used, it can be 'zero' or 'abs' or another floats

    Returns:
    -------
    proximity_Graph : nx.Graph
        the graph with the edges filtered in base of the threshold appllied
    '''
    proximity_Graph = base_Graph.copy()
    if isinstance(threshold, float):
        max = np.max(cut_off_distance, threshold)
        min = np.min(cut_off_distance, threshold)
        interval = range(min, max)
    if isinstance(threshold, str):
        if threshold == 'zero':
            interval = range(0, cut_off_distance)
        if threshold == 'abs':
            interval = range(-np.abs(cut_off_distance),
                             np.abs(cut_off_distance))

    # remove edges:
    for source, target in base_Graph.edges:
        if not base_Graph.get_edge_data(source, target)[feature] in interval:
            proximity_Graph.remove_edge(source, target)
    return proximity_Graph

# create a function to list edges attributes


def get_edge_attribute_list(G: nx.Graph,
                            attribute_to_be_in: str = '',
                            ) -> Tuple[List[str], bool]:
    '''
    the function works to 2 possible verification features: 
    - if the attribute is in the list of attributes of the edges
    - return the list of attributes of edges in the G graph
    Parameters:
    -----------
    G : nx.Graph
        the graph to be considered from which get the edge list
    attribute_to_be_in : str
        the attribute to be checked in the list of attributes of the edges

    Returns:
    --------
    list_of_attributes : list[str,...]
        the list of attributes of the edges in the graph
    is_in : bool
        the boolean value to check if the attribute is in the list of attributes'''
    edge_attribute_list = []
    edge_data = G.edges(data=True)
    for u, v, data in edge_data:
        for attr in data.keys():
            if attr not in edge_attribute_list:
                edge_attribute_list.append(attr)
    is_in = attribute_to_be_in in edge_attribute_list
    return (edge_attribute_list, is_in)


def get_node_atttribute_list(G:  nx.Graph,
                             attribute_to_be_in: str = '',
                             ) -> Tuple[List[str], bool]:
    '''
    the function works to 2 possible verification features:
    - if the attribute is in the list of attributes of the nodes
    - return the list of attributes of nodes in the G graph
    Parameters:
    -----------
    G : nx.Graph
        the graph to be considered from which get the node list
    attribute_to_be_in : str
        the attribute to be checked in the list of attributes of the nodes

    Returns:
    --------
    list_of_attributes : list[str,...]
        the list of attributes of the nodes in the graph

    is_in : bool
        the boolean value to check if the attribute is in the list of attributes'''
    _, feature_dict = list(G.nodes(data=True))[0]
    if attribute_to_be_in != '':
        is_in = attribute_to_be_in in list(feature_dict.keys())
    else:
        is_in = True
    return (list(feature_dict.keys()), is_in)


def weight_on_edge(contact: float = 0,
                   lenght: float = 1,
                   instability: float = 0,
                   ) -> dict:
    """
    it works with the weight on the edge, in case a linear combination on edge for modularity is required

    Parameters: 
    ----------
    contact: float
        the weight of the contact
    lenght: float
        the weight of the lenght
    instability: float
        the weight of the instability

    Returns:
    -------
    weight_dict: dict
        the dictionary containing the weights
    """
    normalized_weight = np.sum([contact, lenght, instability])
    weight_dict = {'contact_in_sequence': contact/normalized_weight, 'lenght': lenght /
                   normalized_weight, 'instability': instability/normalized_weight}
    return weight_dict

# NOTE not necessary anymore


def resolution_respecting_the_kmeans(kmeans_labels: dict | int,
                                     n_ground_cluster: int | pd.Series | list = None
                                     ) -> int:
    """
    this function compute an approximate calculus of resolution

    Parameters:
    ----------
    kmeans_labels: dict | int 
        the dictionary containing the labels of the clusters
        #NOTE it can be also the len of web groups in dataframew

    Returns:
    -------
    resolution: int
        the resolution of the partition expected to be
    """
    if n_ground_cluster is None or not n_ground_cluster:
        n_clusters = 4
    if n_ground_cluster and not isinstance(n_ground_cluster, int):
        n_clusters = set(n_ground_cluster)
        if len(n_clusters) == len(n_ground_cluster):
            raise ValueError(
                'the ground_cluster considered is inappropriate for the analysis')
    elif isinstance(n_ground_cluster, int):
        n_clusters = n_ground_cluster
    if isinstance(kmeans_labels, dict):
        n_cluster_in_graph = set([kmeans_labels.values()])
    elif isinstance(kmeans_labels, int):
        n_cluster_in_graph = kmeans_labels

    resolution = len(n_cluster_in_graph)/(n_clusters)
    return resolution


def add_weight_combination(G: nx.Graph,
                           weight_to_edge: dict
                           ) -> nx.Graph:
    '''
    it give a list of weight to use for louvain partitions

    Parameters:
    ------------
    G : networkx.Graph
        the graph to partition
    weight_to_edge : a dict containing as key the name of edge attributes, as value the weight to associate to it

    Return:
    H : nx.Graph
        a networkxGraph with edge attribution weight obtained as a linear combination of input tuple
        the attribute for edge is weight_combination

    '''
    # first check if the attributes in weight_to_edge are in list_of_attributes:
    list_of_attributes = set()
    for *_, d in G.edges(data=True):
        list_of_attributes.update(d.keys())

    for key in weight_to_edge.keys():
        if str(key) not in list_of_attributes:
            weight_to_edge[key] = 0.
            logging.error('the attribute {0} is not in the list of attributes of the graph'.format(
                key))
    if True not in [bool(val) for val in weight_to_edge.values()]:
        raise AttributeError(
            'there are no compatibily feature in the dictionary you use')

    for u, v, edge in G.edges(data=True):
        weight_sum = 0
        for key in weight_to_edge.keys():
            weight_sum += float(edge[key])*float(weight_to_edge[key])
        edge['weight_combination'] = weight_sum

    return G


def add_louvain_community_attribute(G: nx.Graph,
                                    weight_of_edges: str,  # it has to be the edge attribute
                                    resolution: float  # to define granularity
                                    ) -> tuple[nx.Graph, dict]:
    '''
    adds the attribute to the nodes respecting the louvain community

    Parameters: 
    ------------
    G : nx.Graph
        the graph whose partitions has to be calculate
    weight_of_edges : str
        the attribute of the edges to consider, for modularity calculi
    resolution : float
        if resolution<1 : prefer greater communities
        if resolution>1 : prefer smaller communities

    Returns:
    --------
    H : nx.Graph
        the graph with the community attribute added to the nodes
    community_mapping : dict
        the dictionary containing the mapping between nodes and communities obtained by louvain community method
    '''
    list_of_attributes = set()
    for *_, d in G.edges(data=True):
        list_of_attributes.update(d.keys())

    if weight_of_edges not in list_of_attributes:
        raise AttributeError(
            'the attribute {0} is not in the list of attributes of edges in this graph'.format(weight_of_edges))

    # create partitions and the dictionary to add teh corresponding attribute on each node of the graph
    partitions = nx.community.louvain_communities(
        G, weight=weight_of_edges, resolution=resolution)

    community_mapping = {}
    for community, group_of_nodes in enumerate(partitions):
        for node in group_of_nodes:
            community_mapping[node] = community

    # add the attribute:
    for node, community in community_mapping.items():
        G.nodes[node]['louvain_community'] = community

    return (G,  community_mapping)
