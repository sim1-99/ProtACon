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

    Parameters:
    ----------
    dataframe_of_features: pd.DataFrame
        the dataframe containing the features of the aminoacids
    edges_weight_list: list[tuple[str, str, float, float, bool]] | list
        the list of the edges with their features expressed in floats or bool

    """
    feature_to_be_in = []
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
        Completed_Graph_AAs.add_node(row['AA_pos'])

    # FIXME filter information using the format df[['columns to be in the dict' ]]
    node_attributes_dict = df_x_graph.to_dict(orient='index')
    nx.set_node_attributes(Completed_Graph_AAs, values=node_attributes_dict)

    for edge, distance, instability, in_contact in edges_weight_list:
        source, target = edge

        Completed_Graph_AAs.add_edge(
            *edge, lenght=distance, stability=-instability, contact_in_sequence=in_contact)

    return Completed_Graph_AAs
