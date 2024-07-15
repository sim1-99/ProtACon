#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import numpy as np
import pandas as pd
import networkx as nx

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


def get_the_complete_Graph(dataframe_of_features: pd.DataFrame,
                           edges_weight_list: list[tuple[str, str, float, float, bool]] | list,
                           ) -> nx.Graph:
    """
    this function create a complete graph assigning both to the edges 
    and the nodes some attributes, depending the feature present in the dataframe_of_features and the edges_weight_list
    # FIXME add feature in the dataframe
    Parameters:
    ----------
    dataframe_of_features: pd.DataFrame
        the dataframe containing the features of the aminoacids
    edges_weight_list: list[tuple[str, str, float, float, bool]] | list
        the list of the edges with their features expressed in floats or bool

    """
    feature_to_be_in = ['AA_Name', 'AA_Coords', 'AA_Hydropathy', 'AA_Volume', 'AA_Charge', 'AA_PH', 'AA_iso_PH', 'AA_Hydrophilicity', 'AA_Surface_accessibility',
                        'AA_ja_transfer_energy_scale', 'AA_self_Flex', 'AA_local_flexibility', 'AA_secondary_structure', 'AA_aromaticity', 'AA_human_essentiality']
    for feat in feature_to_be_in:
        if not feat in dataframe_of_features.columns:
            raise ValueError(
                f'feature {feat} is not in the dataframe\nbe sure to use the correct df')
    if 'AA_pos' in dataframe_of_features.columns:
        df_x_graph = dataframe_of_features.set_index('AA_pos')
    else:
        if dataframe_of_features.index.name == 'AA_pos':
            df_x_graph = dataframe_of_features
        else:
            raise ValueError(
                'AA_pos is not in the dataframe, unable to label the nodes in the Graph')

    Completed_Graph_AAs = nx.Graph()
    for _, row in dataframe_of_features.iterrows():
        Completed_Graph_AAs.add_node(row['AA_pos'])[['AA_Name', 'AA_Coords', 'AA_Hydropathy', 'AA_Volume', 'AA_Charge', 'AA_PH', 'AA_iso_PH', 'AA_Hydrophilicity',
                                                     'AA_Surface_accessibility', 'AA_ja_transfer_energy_scale', 'AA_self_Flex', 'AA_local_flexibility', 'AA_secondary_structure', 'AA_aromaticity', 'AA_human_essentiality']]

    node_attributes_dict = df_x_graph.to_dict(orient='index')
    nx.set_node_attributes(Completed_Graph_AAs, values=node_attributes_dict)

    for edge, distance, instability, in_contact in edges_weight_list:
        source, target = edge
        if not source in Completed_Graph_AAs.nodes:
            raise ValueError(
                f'the {source} the is not in the nodes of the Graph')
        if not target in Completed_Graph_AAs.nodes:
            raise ValueError(
                f'the {target} the is not in the nodes of the Graph')
        Completed_Graph_AAs.add_edge(
            *edge, lenght=distance, stability=-instability, contact_in_sequence=in_contact)

    return Completed_Graph_AAs


def compute_proximity_Graph(base_Graph: nx.Graph,
                            cut_off_distance: float,  # use the cut off of config.txt as default
                            feature: str = 'lenght',
                            threshold: str = 'zero' | float
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
