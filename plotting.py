#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting.

This script plots, given one peptide chain:
    1. the distance map between each couple of amino acids
    2. the normalized contact map between each couple of amino acids
    3. the binary thresholded contact map between each couple of amino acids
    4. the attention masks from each head of the last layer
    5. the averages of the attention masks independently computed for each
    layer
    6. the average of the layer attention averages, relative to the whole model
    7. the heatmaps of the absolute attention given to each amino acid by each
    attention head
    8. the heatmaps of the relative attention in percentage given to each amino
    acid by each attention head
    9. the heatmaps of the relative attention in percentage given to each amino
    acid by each attention head, but weighted by the number of occurrencies of
    the corresponding amino acid in the peptide chain
    10. the heatmap of the attention similarity between each couple of amino
    acids
    11. the heatmap of the attention alignment of each head
    12. the bar plot of the attention alignment of each layer
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

import logging
from pathlib import PosixPath

import config_parser
from modules.plot_functions import (
    find_best_nrows,
    plot_attention_masks,
    plot_attention_to_amino_acids,
    plot_bars,
    plot_distance_and_contact,
    plot_heatmap
    )
from modules.utils import Loading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def main(distance_map: np.ndarray,
         norm_contact_map: np.ndarray,
         binary_contact_map: np.ndarray,
         attention: tuple[torch.Tensor],
         attention_per_layer: tuple[torch.Tensor],
         model_attention_average: torch.Tensor,
         attention_to_amino_acids: tuple[torch.Tensor],
         attention_sim_df: pd.DataFrame,
         attention_alignment: tuple[np.ndarray],
         seq_dir: PosixPath,
         types_of_amino_acids: list[str]) -> None:
    """
    Plot and save to seq_dir the arguments received.

    Parameters
    ----------
    distance_map : np.ndarray
        shows the distance - expressed in Angstroms - between each couple of
        amino acids in the peptide chain
    norm_contact_map : np.ndarray
        shows how much each amino acid is close to all the others, in a
        scale between 0 and 1
    binary_contact_map : np.ndarray
        contact map binarized using two thresholding criteria
    attention : tuple
        contains tensors that store the attention from the model, cleared of
        the attention relative to tokens [CLS] and [SEP]
    attention_per_layer : tuple
        averages of the attention masks in each layer
    model_attention_average : torch.Tensor
        average of the average attention masks per layer
    attention_to_amino_acids : tuple
        contains three torch tensors having dimension (number_of_amino_acids,
        number_of_layers, number_of_heads), respectively storing the absolute,
        the relative and the weighted attention given to each amino acid by
        each attention head
    attention_sim_df : pd.DataFrame
        stores attention similarity between each couple of amino acids
    attention_alignment : tuple
        contains two numpy arrays, respectively storing how much attention
        aligns with indicator_function for each attention masks and for each
        average attention mask computed independently over each layers
    seq_dir : PosixPath
        path to the folder containing the plots relative to the peptide chain
    types_of_amino_acids : list
        contains strings with single letter amino acid codes of the amino acid
        types in the peptide chain

    Returns
    -------
    None.

    """
    config = config_parser.Config("config.txt")

    cutoffs = config.get_cutoffs()
    distance_cutoff = cutoffs["DISTANCE_CUTOFF"]
    position_cutoff = cutoffs["POSITION_CUTOFF"]

    nrows = find_best_nrows(len(types_of_amino_acids))
    seq_ID = seq_dir.stem

    # 1-2
    logging.info("Plots 1-2")
    with Loading("Plotting distance and contact maps"):
        plot_distance_and_contact(distance_map, norm_contact_map, seq_dir)
    # 3
    with Loading("Plotting binary contact map"):
        plot_path = seq_dir/f"{seq_ID}_binary_contact_map.png"
        if plot_path.is_file() is False:
            fig, ax = plt.subplots()
            ax.set_title(
                f"{seq_ID}\nBinary Contact Map - Cutoff {distance_cutoff} Å\n"
                f"Excluding Contacts within {position_cutoff} Positions")
            ax.imshow(binary_contact_map, cmap='Blues')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
    # 4
    with Loading("Plotting attention masks"):
        plot_attention_masks(attention,
                             plot_title="{seq_ID}\nAttention Masks - "
                             "Layer {layer_number}".format(seq_ID=seq_ID,
                                                           layer_number=30))
    # 5
    with Loading("Plotting attention mask averages per layer"):
        plot_attention_masks(attention_per_layer,
                             plot_title=f"{seq_ID}\n"
                             "Averages of the Attention Masks per Layer")
    # 6
    with Loading("Plotting attention mask average over the whole model"):
        plot_attention_masks(model_attention_average,
                             plot_title=f"{seq_ID}\nAverage of the Attention "
                             "Masks over the whole model")
    # 7
    with Loading("Plotting attention to amino acids"):
        plot_attention_to_amino_acids(attention_to_amino_acids[0],
                                      types_of_amino_acids,
                                      plot_title=f"{seq_ID}\n"
                                      "Attention to Amino Acids")
    # 8
    with Loading("Plotting relative attention to amino acids in percentage"):
        plot_attention_to_amino_acids(attention_to_amino_acids[1],
                                      types_of_amino_acids,
                                      plot_title=f"{seq_ID}\nRelative "
                                      "Attention to Amino Acids in Percentage")
    # 9
    with Loading("Plotting weighted attention to amino acids in percentage"):
        plot_attention_to_amino_acids(attention_to_amino_acids[2],
                                      types_of_amino_acids,
                                      plot_title=f"{seq_ID}\nWeighted "
                                      "Attention to Amino Acids in Percentage")
    # 10
    with Loading("Plotting attention similarity"):
        plot_heatmap(attention_sim_df,
                     plot_title=f"{seq_ID}\nPairwise Attention Similarity - "
                     "Pearson Correlation")
    # 11
    with Loading("Plotting attention alignment"):
        plot_heatmap(attention_alignment[0],
                     plot_title=f"{seq_ID}\nAttention Alignment")
    # 12
    with Loading("Plotting attention alignment per layer"):
        plot_bars(attention_alignment[1],
                  plot_title=f"{seq_ID}\nAttention Alignment per Layer")

    plt.close('all')
