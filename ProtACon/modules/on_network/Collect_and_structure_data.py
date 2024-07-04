#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

from miscellaneous import get_AA_features_dataframe, CA_Atom
from modules import miscellaneous
from ProtACon import run_protbert
import pandas as pd
import numpy as np

# NOTE put the dataframe generation in an unique function to return a tuple of dataframe, an original one, a filtred for pca and another for the network

# AA_dataframe = get_AA_features_dataframe(CA_Atoms)
features_dataframe_columns = ('AA_Name', 'AA_Coords', 'AA_Hydropathy', 'AA_Volume', 'AA_Charge_Density', 'AA_RCharge_density',
                              'AA_Charge', 'AA_PH', 'AA_iso_PH', 'AA_Hydrophilicity', 'AA_Surface_accessibility',
                              'AA_ja_transfer_energy_scale', 'AA_self_Flex', 'AA_local_flexibility', 'AA_secondary_structure',
                              'AA_aromaticity', 'AA_human_essentiality', 'AA_web_group')


def get_dataframe_for_PCA(base_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Starting from the general feature dataframe, it removes the columns of data corresponding to the not physio-chemical information
    like the coords or similar

    Parameters:
    base_features_df : get_AA_features_dataframe(CA_Atoms : tuple[CA_Atom, ...]) type of pd.DataFrame
        The general feature dataframe, containing info on:
        - AA_Name
        - AA_Coords
        - AA_Hydropathy
        - AA_Volume
        - AA_Charge_Density
        - AA_RCharge_density
        - AA_Charge
        - AA_PH
        - AA_iso_PH
        - AA_Hydrophilicity
        - AA_Surface_accessibility
        - AA_ja_transfer_energy_scale
        - AA_self_Flex
        - AA_local_flexibility
        - AA_secondary_structure
        - AA_aromaticity
        - AA_human_essentiality
        - AA_web_group

    Returns:
    dataframe_pca : pd.DataFrame
        The dataframe prepared for pca computing to get the data to reduce on, it has to contain the info on:
        - AA_Hydropathy
        - AA_Volume
        - AA_Charge
        - AA_PH
        - AA_iso_PH
        - AA_Hydrophilicity
        - AA_Surface_accessibility
        - AA_ja_transfer_energy_scale
        - AA_self_Flex
        - AA_local_flexibility
        - AA_secondary_structure
        - AA_aromaticity
        - AA_human_essentiality

    """
    features_dataframe_columns = ('AA_Name', 'AA_Coords', 'AA_Hydropathy', 'AA_Volume', 'AA_Charge_Density', 'AA_RCharge_density',
                                  'AA_Charge', 'AA_PH', 'AA_iso_PH', 'AA_Hydrophilicity', 'AA_Surface_accessibility',
                                  'AA_ja_transfer_energy_scale', 'AA_self_Flex', 'AA_local_flexibility', 'AA_secondary_structure',
                                  'AA_aromaticity', 'AA_human_essentiality', 'AA_web_group')

    dataframe_pca = base_features_df.copy()
    for feature in features_dataframe_columns:
        if feature not in base_features_df.columns:
            raise ValueError(
                f'The feature {feature} is not present in the dataframe,\ncheck the correct general feature dataframe to get the df for PCA analysis')
    dataframe_pca = dataframe_pca.drop(columns={
                                       'AA_Coords', 'AA_Charge_Density', 'AA_RCharge_density', 'AA_web_group'}, inplace=True)

    dataframe_pca['AA_Name'] = dataframe_pca['AA_Name'] + \
        '(' + str(dataframe_pca.index) + ')'

    dataframe_pca.set_index('AA_Name', inplace=True)

    return dataframe_pca


def generate_index_df(CA_Atoms: tuple[CA_Atom, ...] | False,
                      column_of_df: pd.Series | False
                      ) -> tuple[str, ...] | pd.Series:
    """
    Generate the index for dataframe, to use them as label of nodes:
    The index is generated as the AA_Name + the index of the AA in the peptide chain
    Or it use a s base a column of a dataframe

    Parameters:
    -----------
    CA_Atoms : tuple[CA_Atom, ...]
        The tuple of CA_Atom objects

    column_of_df : pd.Series
        The column of the dataframe to use as index reference

    Returns:
    --------
    index_tuple : tuple[str,...]
        The index to use as label of nodes, or to return as a pd.Series to next set as index
    """
    if not CA_Atoms and not column_of_df:
        raise ValueError(
            'You must provide up to one of the two parameters to generate the index')
    elif CA_Atom:
        index_tuple = tuple(
            [atom.AA_Name + '(' + str(atom.idx) + ')' for atom in CA_Atoms])
    elif column_of_df:
        for element in list(column_of_df):
            try:
                index_tuple = tuple(
                    [element + '(' + str(idx) + ')' for idx, element in enumerate(column_of_df)])
            except Exception('unable to perform the operation'):
                index_tuple = column_of_df
    else:
        raise ValueError(
            'You must provide at least one of the two parameters to generate the index')

    return index_tuple


def get_dataframe_from_nparray(base_map: np.ndarray,
                               index_str: tuple[str, ...],
                               columns_str: tuple[str, ...]
                               ) -> pd.DataFrame:
    """
    Generate the dataframe from data stored in a np.ndarray, it works for relationship between amminoacids
    so the dataframe has a double indexing, the same both for index and for columns

    Parameters:
    -----------
    instability_map : np.ndarray
        The instability map to convert in a dataframe

    index_str: tuple[str,...]
        The indices get from generate_index_df function

    columns_str : tuple[str,...]
        The indices get from generate_index_df function
    """
    condition_rows = base_map.shape[0] == len(index_str)
    condition_columns = base_map.shape[1] == len(columns_str)
    if not condition_rows or not condition_columns:
        raise ValueError(
            'The shape of the base_map must be equal to the length of the index and columns')

    instability_df = pd.DataFrame(
        data=base_map, index=index_str, columns=columns_str)
    return instability_df


def get_dataframe_for_network(base_features_df: pd.DataFrame,  # the basic dataframe of all feature from which filter informations
                              # a control param to set new indices through generate_index_df
                              set_new_index: bool = False
                              ) -> pd.DataFrame:
    """
    Generate the DataFrame for constructing the network

    Parameters:
    -----------
    base_features_df : pd.DataFrame
        The general feature dataframe, containing info on:
        - AA_Name
        - AA_Coords
        - AA_Hydropathy
        - AA_Volume
        - AA_Charge_Density
        - AA_RCharge_density
        - AA_Charge
        - AA_PH
        - AA_iso_PH
        - AA_Hydrophilicity
        - AA_Surface_accessibility
        - AA_ja_transfer_energy_scale
        - AA_self_Flex
        - AA_local_flexibility
        - AA_secondary_structure
        - AA_aromaticity
        - AA_human_essentiality
        - AA_web_group

    Returns:
    --------
    dataframe_network : pd.DataFrame
        The dataframe prepared for network construction, containing info on:
        - AA_Name
        - AA_Hydropathy
        - AA_Volume
        - AA_Charge
        - AA_PH
        - AA_iso_PH
        - AA_Hydrophilicity
        - AA_Surface_accessibility
        - AA_ja_transfer_energy_scale
        - AA_self_Flex
        - AA_local_flexibility
        - AA_secondary_structure
        - AA_aromaticity
        - AA_human_essentiality



    """
    features_dataframe_columns = ('AA_Name', 'AA_Coords', 'AA_Hydropathy', 'AA_Volume', 'AA_Charge_Density', 'AA_RCharge_density',
                                  'AA_Charge', 'AA_PH', 'AA_iso_PH', 'AA_Hydrophilicity', 'AA_Surface_accessibility',
                                  'AA_ja_transfer_energy_scale', 'AA_self_Flex', 'AA_local_flexibility', 'AA_secondary_structure',
                                  'AA_aromaticity', 'AA_human_essentiality', 'AA_web_group')

    dataframe_network = base_features_df.copy()
    for feature in features_dataframe_columns:
        if feature not in base_features_df.columns:
            raise ValueError(
                f'The feature {feature} is not present in the dataframe,\ncheck the correct general feature dataframe to get the df for network analysis')
    dataframe_network = dataframe_network.drop(columns={
        'AA_Coords', 'AA_Charge_Density', 'AA_RCharge_density', 'AA_web_group'}, inplace=True)
    if set_new_index:
        dataframe_network['AA_pos'] = generate_index_df(
            dataframe_network['AA_Name'])
        dataframe_network.set_index('AA_pos', inplace=True)

    return dataframe_network


# NOTE add function to get list of edges
def get_list_of_edges(base_map: np.ndarray,
                      CA_Atoms: tuple[CA_Atom, ...]
                      ) -> list[tuple[str, str]]:
    """
    To obtain the list of edges as the source of the nx.Graph:

    Parameters:
    -----------
    base_map : np.ndarray
        The binary map from which get the edge, as a couple of coords

    CA_Atoms : tuple[CA_Atom, ...]
        The tuple of CA_Atom objects from which get the name of the nodes: the labels of the nx

    Returns:
    --------
    list_of_edges : list[tuple[str, str]]
        The list of edges to use as source of the nx.Graph, as a list of couples (source, target)  

    """
    list_of_edges = []
    indices = generate_index_df(CA_Atoms)

    df = pd.DataFrame(base_map, index=indices, columns=indices)
    for i in range(df.axes[0]):
        for j in range(df.axes[1]):
            content = df.iloc[i, j]
            if content == 1:
                list_of_edges.append(df.index[i], df.columns[j])

    return list_of_edges


def get_weight_for_edges(list_of_edges: list[tuple[str, str]],
                         base_df: pd.DataFrame,
                         instability_df: pd.DataFrame,
                         CA_Atoms: tuple[CA_Atom, ...]
                         ) -> list[tuple[str, str, float, float, bool]]:
    """
    To obtain the list of edges with weight associated to them
    Parameters:
    -----------
    list_of_edges : list[tuple[str, str]]
        The list of edges as a list of couples (source, target)

    base_df : np.ndarray
        The contact map from which take in consideration only the link between peptides next to eachother

    instability_df : np.ndarray
        The instability map from which take the instability of the link between peptides next to eachother

    CA_Atoms : tuple[CA_Atom, ...]
        The tuple of CA_Atom objects from which get the name of the nodes: the labels of the nx

    Returns:
    --------
    list_of_edges_and_weights : list[tuple[str, str, float, float, bool]]
        The list of edges to use as source of the nx.Graph, as a tuple ((source, target), content)  
        where content is a tuple of weights: distance, instability, contact_in_sequence

    """

    condition_idx = base_df.index == instability_df.index
    condition_columns = base_df.columns == instability_df.columns
    if (not condition_idx or not condition_columns):
        raise ValueError(
            'The index and columns of the two dataframe must be the same')
    list_of_edges_and_weights = []

    for n_row,  row in enumerate(base_df.index):
        for n_column, column in enumerate(base_df.columns):
            if (str(row), str(column)) in list_of_edges:
                if abs(n_row - n_column) == 1:
                    list_of_edges_and_weights.append(
                        ((row, column), base_df.at[row, column], instability_df.at[row, column], True))
                else:
                    list_of_edges_and_weights.append(
                        ((row, column), base_df.at[row, column], instability_df.at[row, column], False))

    return list_of_edges_and_weights

#
