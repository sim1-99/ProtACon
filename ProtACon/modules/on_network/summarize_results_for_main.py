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

from ProtACon.modules.on_network import networks_analysis, kmeans_computing_and_results, PCA_computing_and_results, Attention_map_from_networks
from ProtACon.modules.miscellaneous import CA_Atom, get_AA_features_dataframe
import pandas as pd

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


# Define the results for louvain
