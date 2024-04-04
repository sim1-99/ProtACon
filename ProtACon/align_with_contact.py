#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align with contact.

This script combines other scripts for the computation of the attention
alignment of the contact map of one protein. Other meaningful quantities, such
as pairwise attention similarity, are computed too. In case of a set of
proteins, those quantities can be averaged over it. The user can also choose if
to plot and save all the plots of every single protein in the set.
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from IPython.display import display
from pathlib import Path
import warnings

from ProtACon import config_parser
from ProtACon.modules.attention import clean_attention
from ProtACon.modules.miscellaneous import (
    get_model_structure,
    get_types_of_amino_acids
    )
from ProtACon.modules.plot_functions import plot_bars, plot_heatmap
from ProtACon.modules.utils import average_maps_together, Loading

from ProtACon import run_protbert
from ProtACon import preprocess_attention
from ProtACon import process_attention
from ProtACon import process_contact
from ProtACon import plotting

import numpy as np
import pandas as pd
import torch


config = config_parser.Config("config.txt")

paths = config.get_paths()
plot_folder = paths["PLOT_FOLDER"]
plot_dir = Path(__file__).resolve().parents[1]/plot_folder


def main(
        seq_ID: str,
        save_single=False
        ) -> (
            torch.Tensor,
            pd.DataFrame,
            np.ndarray,
            np.ndarray
            ):
    """
    Run the main function of align_with_contact.py.

    It computes the attention alignment with the contact map and other
    quantities for the peptide chain identified with seq_ID.

    Parameters
    ----------
    seq_ID : str
        alphanumerical code representing uniquely one peptide chain
    save_single : bool, default is False
        if True, run plotting.main() and save the plots

    Returns
    -------
    att_sim_df : pd.DataFrame
        stores attention similarity between each couple of amino acids
    head_att_align : np.ndarray
        array having dimension (number_of_layers, number_of_heads), storing how
        much attention aligns with indicator_function for each attention masks
    layer_att_align : np.ndarray
        array having dimension (number_of_layers), storing how much attention
        aligns with indicator_function for each average attention mask computed
        independently over each layer

    """
    seq_dir = plot_dir/seq_ID
    seq_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_attention, raw_tokens, CA_Atoms = run_protbert.main(seq_ID)

    attention = clean_attention(raw_attention)
    tokens = raw_tokens[1:-1]
    number_of_heads, number_of_layers = get_model_structure(raw_attention)
    types_of_amino_acids = get_types_of_amino_acids(tokens)

    return_preprocess_attention = preprocess_attention.main(
        attention, tokens, seq_dir)

    display(return_preprocess_attention[0])
    types_of_amino_acids.sort()

    distance_map, norm_contact_map, binary_contact_map = process_contact.main(
        CA_Atoms)

    return_process_attention = process_attention.main(
        attention, return_preprocess_attention[1], binary_contact_map,
        types_of_amino_acids)

    attention_to_amino_acids = (return_preprocess_attention[1],
                                return_preprocess_attention[2],
                                return_preprocess_attention[3])

    attention_averages = (return_process_attention[1],
                          return_process_attention[2])

    attention_alignment = (return_process_attention[3],
                           return_process_attention[4])

    if save_single is True:
        plotting.main(
            distance_map, norm_contact_map, binary_contact_map, attention,
            attention_averages, attention_to_amino_acids,
            return_process_attention[0], attention_alignment, seq_dir,
            types_of_amino_acids)

        return None, None, None

    return (
        return_process_attention[0],
        attention_alignment[0],
        attention_alignment[1]
        )


def average_on_set(
        att_sim_df_list: list[pd.DataFrame],
        head_att_align_list: list[np.ndarray],
        layer_att_align_list: list[np.ndarray]
        ) -> (
            pd.DataFrame,
            np.ndarray,
            np.ndarray
            ):
    """
    Compute attention alignment and similarity over the whole set of proteins.

    Parameters
    ----------
    att_sim_df_list : list[pd.DataFrame]
        contains the attention similarity between each couple of amino acids
        for each peptide chain
    head_att_align_list : list[np.ndarray]
        contains one array for each peptide chain, each one having dimension
        (number_of_layers, number_of_heads), storing how much attention aligns
        with indicator_function for each attention masks
    layer_att_align_list : list[np.ndarray]
        contains one array for each peptide chain, each one having dimension
        (number_of_layers), storing how much attention aligns with
        indicator_function for each average attention mask computed
        independently over each layer

    Returns
    -------
    avg_att_sim_df : pd.DataFrame
        attention similarity averaged over the whole protein set
    avg_head_att_align : np.ndarray
        head attention alignment averaged over the whole  protein set
    avg_layer_att_align : np.ndarray
        layer attention alignment averaged over the whole  protein set

    """
    with Loading("Computing average attention similarity"):
        avg_att_sim_df = average_maps_together(att_sim_df_list)

    avg_att_sim_df.to_csv(
        plot_dir/"attention_sim_df.csv", index=True, sep=';')

    with Loading("Computing average head attention alignment"):
        avg_head_att_align = average_maps_together(head_att_align_list)

    with Loading("Computing average layer attention alignment"):
        avg_layer_att_align = average_maps_together(layer_att_align_list)

    return (avg_att_sim_df,
            avg_head_att_align,
            avg_layer_att_align
            )


def plot_average_on_set(
        avg_att_sim_df: pd.DataFrame,
        avg_head_att_align: np.ndarray,
        avg_layer_att_align: np.ndarray
        ) -> None:
    """
    Plot attention alignment and similarity over the whole set of proteins.

    Parameters
    ----------
    avg_att_sim_df : pd.DataFrame
        attention similarity averaged over the whole protein set
    avg_head_att_align : np.ndarray
        head attention alignment averaged over the whole  protein set
    avg_layer_att_align : np.ndarray
        layer attention alignment averaged over the whole  protein set

    Returns
    -------
    None.

    """
    with Loading("Plotting average attention similarity"):
        plot_heatmap(avg_att_sim_df,
                     plot_title="Average Pairwise Attention Similarity\n"
                     "Pearson Correlation")

    with Loading("Plotting average head attention alignment"):
        plot_heatmap(avg_head_att_align,
                     plot_title="Average Head Attention Alignment")

    with Loading("Plotting average layer attention alignment"):
        plot_bars(avg_layer_att_align,
                  plot_title="Average Layer Attention Alignment")
