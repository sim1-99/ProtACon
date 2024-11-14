#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils.ProtParamData import DIWV

from ProtACon.modules.basics import (
    CA_Atom,
    extract_CA_Atoms,
    get_AA_features_dataframe,
    protein_reference_point,
)
from ProtACon import config_parser

# NOTE put the dataframe generation in an unique function to return a tuple of dataframe, an original one, a filtred for pca and another for the network
# FIXME  adjust the dataframe compute and use only one, add and remove columns only when needed
# AA_dataframe = get_AA_features_dataframe(CA_Atoms)

# classe protina usa miscellaneous.get_sequence_to_tokenize
class Protein_id:
    def __init__(
            self,
            name_id: str,
            first_ten_attention_score: dict[tuple, float] = {},

    ):
        self.name_id = name_id
        self.first_ten_attention_score = first_ten_attention_score

    def extract_bio_features(
            self,
    ) -> dict:
        config_file_path = Path(__file__).resolve().parents[3]/"config.txt"
        config = config_parser.Config(config_file_path)

        paths = config.get_paths()
        pdb_folder = paths["PDB_FOLDER"]
        pdb_dir = Path(__file__).resolve().parents[1]/pdb_folder
        file_path = pdb_dir/f"pdb{self.name_id.lower()}.ent"

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            structure = PDBParser().get_structure(self.name_id, file_path)
        CA_Atoms = extract_CA_Atoms(structure)
        protein_dict = protein_reference_point(CA_Atoms=CA_Atoms)
        return protein_dict


def generate_index_df(CA_Atoms: tuple[CA_Atom, ...] = (),
                      column_of_df: pd.Series = [],
                      ) -> tuple[str, ...] | pd.Series:
    """
    Generate the index for dataframe, to use them as label of nodes:
    The index is generated as the AA_Name + the index of the AA in the peptide chain
    Or it use a s base a column of a dataframe in format AA(#idx)

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
    if not len(CA_Atoms) and not column_of_df.any():
        raise ValueError(
            'You must provide up to one of the two parameters to generate the index')
    # if given only the list of atoms
    elif len(CA_Atoms) > 0:
        index_tuple = tuple(
            [atom.name + '(' + str(atom.idx) + ')' for atom in CA_Atoms])
    # if given only the column of the dataframe
    elif column_of_df.any():
        try:
            index_tuple = tuple(
                [element + '(' + str(idx) + ')' for idx, element in enumerate(column_of_df)])
        except Exception('unable to perform the operation'):
            index_tuple = column_of_df
    else:
        raise ValueError(
            'You must provide at least one of the two parameters to generate the index')

    return index_tuple


def get_dataframe_for_PCA(CA_Atoms: tuple[CA_Atom, ...]) -> pd.DataFrame:
    """
    Starting from the general feature dataframe, it removes the columns of data corresponding to the not physio-chemical information
    like the coords or similar

    Parameters:
    base_features_df : get_AA_features_dataframe(CA_Atoms : tuple[CA_Atom, ...]) type of pd.DataFrame
        The general feature dataframe, containing info on:


    """
    dataframe = get_AA_features_dataframe(CA_Atoms=CA_Atoms)
    dataframe_pca = dataframe.copy()
    columns_to_remove = ['AA_Coords', 'AA_Charge_Density',
                         'AA_RCharge_density', 'AA_web_group']
    for column in dataframe_pca.columns:
        if column in columns_to_remove:
            dataframe_pca.drop(columns=column, inplace=True)
    index_label = generate_index_df(CA_Atoms=CA_Atoms)
    dataframe_pca['AA_pos'] = index_label
    dataframe_pca.set_index('AA_pos', inplace=True)
    if 'AA_Name' in dataframe_pca.columns:
        dataframe_pca.drop(columns='AA_Name', inplace=True)
    return dataframe_pca


def get_dataframe_from_nparray(base_map: np.ndarray,
                               index_str: tuple[str, ...],
                               columns_str: tuple[str, ...]
                               ) -> pd.DataFrame:
    """
    Generate the dataframe from data stored in a np.ndarray, it works for relationship between amminoacids
    so the dataframe has a double indexing, the same both for index and for columns

    Parameters:
    -----------
    base_map : np.ndarray
        The map to convert in a dataframe

    index_str: tuple[str,...]
        The indices get from generate_index_df function

    columns_str : tuple[str,...]
        The indices get from generate_index_df function

    Returns:
    --------
    df : pd.DataFrame
        The dataframe obtained from the np.ndarray base_map
        # TODO ? add feature in the dataframe
    """
    condition_rows = base_map.shape[0] == len(index_str)
    condition_columns = base_map.shape[1] == len(columns_str)
    if not condition_rows or not condition_columns:
        raise ValueError(
            'The shape of the base_map must be equal to the length of the index and columns')

    df = pd.DataFrame(
        data=base_map, index=index_str, columns=columns_str)
    return df


def get_df_about_instability(base: pd.DataFrame | tuple[CA_Atom, ...],
                             set_indices: str = False
                             ) -> pd.DataFrame:
    """
    generate the dataframe associated to the one of the base of CA_atoms list 
    of the instability contact between edges:

    Parameters:
    -----------
    base_dataframe : pd.DataFrame
        The dataframe from which take the indices/columns
    # FIXME add feature in the dataframe
    set_indices: str
        The column to use as index of the dataframe

    Returns:
    --------
    df_instability : pd.DataFrame
        The dataframe containing the instability of the contacts between peptides
    """
    if isinstance(base, pd.DataFrame):  # if a dataframe it get the instability df with indices as single letter AA
        if not set_indices:
            list_of_index = base.index
        else:
            list_of_index = base[set_indices]
    else:
        # if given base as the CA_Atoms list ithe instability index has the format AA(#idx)
        list_of_index = generate_index_df(CA_Atoms=base)

    df_instability = pd.DataFrame(index=list_of_index, columns=list_of_index)
    for AA_row in list_of_index:
        for AA_col in list_of_index:
            # to be sure to take the letter only in case of AA(#idx) format
            df_instability.loc[AA_row,
                               AA_col] = DIWV[AA_row[0].upper()][AA_col[0].upper()]
    return df_instability


# NOTE add function to get list of edges
def get_list_of_edges(base_map: np.ndarray,
                      CA_Atoms: tuple[CA_Atom, ...],
                      type: str = 'str',
                      ) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """
    To obtain the list of edges as the source of the nx.Graph:

    Parameters:
    -----------
    base_map : np.ndarray
        The binary map from which get the edge, as a couple of coords

    CA_Atoms : tuple[CA_Atom, ...]
        The tuple of CA_Atom objects from which get the name of the nodes: the labels of the nx
    type : str
        if type is 'str' then the returned list have AA(#idx) format
        otherwise just the #idx format
    Returns:
    --------
    list_of_edges : list[tuple[str, str]]
        The list of edges to use as source of the nx.Graph, as a list of couples (source, target)  
    base_df : pd.DataFrame
        dataframe with values as the base_map : base_df.values() == base_map

    """
    list_of_edges = []
    if type == 'str':
        indices = generate_index_df(CA_Atoms=CA_Atoms)
        base_df = pd.DataFrame(base_map, index=indices, columns=indices)
        for i in range(len(base_df.axes[0])):
            for j in range(len(base_df.axes[1])):
                content = base_df.iloc[i, j]
                if content:
                    list_of_edges.append(
                        (base_df.index[i], base_df.columns[j]))
    elif type == 'int':
        coordinates = np.argwhere(base_map != 0)
        list_of_edges = [tuple(edge) for edge in coordinates]

    return (list_of_edges, base_df)


def get_weight_for_edges(list_of_edges: list[tuple[str, str]],
                         base_df: pd.DataFrame,
                         instability_df: pd.DataFrame,
                         ) -> list[tuple[str, str, float, float, bool]]:
    """
    To obtain the list of edges with weight associated to them starting from a list of edges and the maps from 
    which get the weights
    #Parameters:
    -----------
    list_of_edges : list[tuple[str, str]]
        The list of edges as a list of couples (source, target)

    base_df : pd.DataFrame
        The distancies_df from which take in consideration only the lenght of the link between peptides next to each other

    instability_df : pd.DataFrame
        The instability_df from which take the instability index of the link between peptides next to each other

    CA_Atoms : tuple[CA_Atom, ...]
        The tuple of CA_Atom objects from which get the name of the nodes: the labels of the nx

    #Returns:
    --------
    list_of_edges_and_weights : list[tuple[str, str, float, float, bool]]
        The list of edges to use as source of the nx.Graph, as a tuple ((source, target), content)  
        where content is a tuple of weights: distance, instability, contact_in_sequence

    """

    condition_idx = base_df.index.equals(instability_df.index)
    condition_columns = base_df.columns.equals(instability_df.columns)
    if (not condition_idx or not condition_columns):
        raise ValueError(
            'The index and columns of the two dataframe must be the same')
    list_of_edges_and_weights = []

    for n_row,  row in enumerate(base_df.index):
        for n_column, column in enumerate(base_df.columns):
            if (str(row), str(column)) in list_of_edges or (str(column), str(row)) in list_of_edges:
                if abs(n_row - n_column) == 1:
                    list_of_edges_and_weights.append(
                        ((row, column), base_df.at[row, column], instability_df.at[row, column], True))
                else:
                    list_of_edges_and_weights.append(
                        ((row, column), base_df.at[row, column], instability_df.at[row, column], False))

    return list_of_edges_and_weights


def get_indices_from_str(list_of_edges: list[tuple[str, str]],
                         dataframe_x_conversion: pd.DataFrame,
                         column_containing_key: str
                         ) -> list[tuple[int, int]]:
    """
    it return a conversion from a list of edges expressed in strings, 
    into the respective list of indices associated to each node in the edge,
    to be consider as a key the dataframe and the columns to confront the content to the 
    edge[0], edge [1] to get the indices from.

    Parameters:
    -----------
    list_of_edges : list[tuple[str, str]]
        The list of edges as a list of couples (source, target)

    dataframe_x_conversion : pd.DataFrame
        The dataframe used as conversion table from string to index

    column_containing_key : str
        The column to watch in to search for the index of the key to convert

    Returns:
    --------
    indices_list : list[tuple[int, int]]
        The list of indices associated to each node in the edge
    """
    indices_list = []
    if not column_containing_key in dataframe_x_conversion.columns:
        raise ValueError(
            'The column_containing_key must be present in the dataframe_x_conversion')
    for edge in list_of_edges:
        source_idx = dataframe_x_conversion[dataframe_x_conversion[column_containing_key]
                                            == edge[0]].index[0]
        target_idx = dataframe_x_conversion[dataframe_x_conversion[column_containing_key]
                                            == edge[1]].index[0]
        indices_list.append((source_idx, target_idx))
    return indices_list


def get_the_Graph_network(CA_Atoms: tuple[CA_Atom, ...],
                          edges_weight_list: list[tuple[str, str, float, float, bool]] | list,
                          ) -> tuple[nx.Graph, float]:
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

    # set the indices to name the nodes
    dataframe_of_features = get_AA_features_dataframe(CA_Atoms)
    dataframe_of_features['AA_pos'] = generate_index_df(CA_Atoms=CA_Atoms)
    df_x_graph = dataframe_of_features.set_index('AA_pos')

    # remove the web group column after gettting info on resolution
    columns_to_remove = ['AA_web_group']
    resolution = len(set(df_x_graph.AA_web_group)) / 4.0
    for column in columns_to_remove:
        if column in df_x_graph.columns:
            df_x_graph.drop(columns=column, inplace=True)

    Completed_Graph_AAs = nx.Graph()
    for _, row in dataframe_of_features.iterrows():
        Completed_Graph_AAs.add_node(row['AA_pos'])

    # use [[filtred_cols,....]] to filter columns to get attributes, from
    node_attributes_dict = df_x_graph.to_dict(orient='index')
    nx.set_node_attributes(Completed_Graph_AAs, values=node_attributes_dict)

    for edge, distance, instability, in_contact in edges_weight_list:
        source, target = edge
        if not (source in list(Completed_Graph_AAs.nodes())):
            raise ValueError(
                f'the {source} the is not in the nodes {Completed_Graph_AAs.nodes()} of the Graph')
        if not (target in list(Completed_Graph_AAs.nodes())):
            raise ValueError(
                f'the {target} the is not in the nodes {Completed_Graph_AAs.nodes()} of the Graph')
        Completed_Graph_AAs.add_edge(
            *edge, lenght=distance, instability=instability, contact_in_sequence=in_contact)

    return (Completed_Graph_AAs, resolution)


def get_10_best_scores(
        avg_att_map: np.ndarray | torch.Tensor,
        n_best: int = 10
) -> dict[tuple, float]:
    '''
    it returns the 10 best scores in the average attention map in form of dictionary
    where keys are tuples of (x, y) coordinates and values are the corresponding scores

    Parameters
    ----------
    avg_att_map : np.ndarray
        the average attention map to get the best scores
    n_best : int
        the number of best scores to get, as default 10

    Returns
    -------
    dict[tuple, float]
        a dictionary where keys are tuples of (h_id, lay_id) coordinates and values are the corresponding scores
    '''
    if isinstance(avg_att_map, torch.Tensor):
        avg_att_map = avg_att_map.numpy()

    flatten_array = avg_att_map.flatten()
    sorted_scores = np.sort(flatten_array)[-n_best:]
    best_scores = {}
    for lay_id in range(avg_att_map.shape[0]):
        for h_id in range(avg_att_map.shape[1]):
            if avg_att_map[lay_id, h_id] in sorted_scores:
                best_scores[(h_id, lay_id)] = avg_att_map[lay_id, h_id]
    return best_scores


def get_the_scores_from_att_align(
        att_align_prot: np.ndarray | torch.Tensor,
        list_of_top_head_scores: list[tuple]
) -> dict[tuple, float]:
    '''
    it returns the scores of the top heads in the attention alignment map of the protein
    where keys are tuples of (x, y) coordinates and values are the corresponding scores
    Parameters
    ----------
    att_align_prot : np.ndarray
        the attention alignment map from which get the scores
    list_of_top_head_scores : list[tuple]
        the list of top head scores to get the scores from the attention alignment map

    Returns
    -------
    dict[tuple, float]
    a dictionary where keys are tuples of (h_id, lay_id) coordinates and values are the corresponding scores
    '''
    if isinstance(att_align_prot, torch.Tensor):
        att_align_prot = att_align_prot.numpy()
    scores_from_att_align = {}
    for head_coords in list_of_top_head_scores:
        lay_id, h_id = head_coords
        scores_from_att_align[head_coords] = att_align_prot[lay_id, h_id]
    return scores_from_att_align
