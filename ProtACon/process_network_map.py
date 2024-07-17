#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Give adjacency matrix get from calculus of clusters or label in network
through the CA_Atoms tuple it give back the adjacency matrix representing the function for alignment

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ProtACon.modules.miscellaneous import CA_Atom, get_AA_features_dataframe
from ProtACon.modules.on_network.Collect_and_structure_data import generate_index_df
from ProtACon.modules.on_network.kmeans_computing_and_results import get_clusters_label, dictionary_from_tuple
from ProtACon import process_contact


__author__ = 'Renato Eliasy'
__email__ = 'renatoeliasy@gmail.com'

# map from kmeans


# map from louvain communities

def process_kmeans_map(CA_Atoms: tuple[CA_Atom, ...]
                       ) -> tuple[np.ndarray, np.ndarray,  tuple[list[str], ...]]:
    """
    the main of this script take the list of residues and perform a kmean cluster 
    and a louvain computing partition respective on dataframe and network
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

    columns_to_pop_x_Kmeans = ['AA_Name', 'AA_Coords',
                               'AA_Rcharge_density', 'AA_Charge_Density']
    AA_df_Kmeans = AA_general_dataframe.drop(columns=columns_to_pop_x_Kmeans)

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
            kmean_contact,
            kmeans_tuple_labels)
# add another function for the plots
    # get the proximity graph:
    # get the complete graph from nx
    # get the proximity as a subgraph
    #


def process_louvain_map(CA_Atoms: tuple[CA_Atom, ...],
                        ) -> tuple[np.ndarray, np.ndarray, tuple[list[str], ...]]:
    pass
