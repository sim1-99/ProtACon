"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Given one peptide chain, plot:
    1.1. the distance map between each couple of amino acids
    1.2. the normalized contact map between each couple of amino acids
    1.3. the binary thresholded contact map between each couple of amino acids
    1.4. the attention matrices from each head of the last layer
    1.5. the averages of the attention matrices independently computed for each
    layer
    1.6. the average of the layer attention averages, relative to the whole
    model
    1.7. the heatmaps of the total attention given to each amino acid
    1.8. the heatmap of the attention similarity between each couple of amino
    acids
    1.9. the heatmap of the attention alignment of each head
    1.10. the bar plot of the attention alignment of each layer
    
Given a set of peptide chains, plot:
    2.1. the heatmaps of the percentage of attention given to each amino acid
    2.2. the heatmaps of the percentage of attention given to each amino acid,
    weighted by the occurrences of that amino acid in all the proteins of the
    set
    2.3. the heatmaps of the percentage of each head's attention given to each
    amino acid
    2.4. the heatmap of the average pairwise attention similarity
    2.5. the heatmap of the average head attention alignment
    2.6. the heatmap of the average layer attention alignment

"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ProtACon import config_parser
from ProtACon.modules.plot_functions import (
    plot_attention_matrices,
    plot_attention_to_amino_acids_alone,
    plot_attention_to_amino_acids_together,
    plot_bars,
    plot_distance_and_contact,
    plot_heatmap,
)
from ProtACon.modules.utils import Loading


def plot_on_chain(
    distance_map: np.ndarray,
    norm_contact_map: np.ndarray,
    binary_contact_map: np.ndarray,
    attention: tuple[torch.Tensor, ...],
    attention_avgs: list[torch.Tensor],
    attention_to_amino_acids: torch.Tensor,
    attention_sim_df: pd.DataFrame,
    head_attention_align: np.ndarray,
    layer_attention_align: np.ndarray,
    seq_dir: Path,
    chain_amino_acids: list[str],
) -> None:
    """
    Plot and save to seq_dir the arguments received.

    Parameters
    ----------
    distance_map : np.ndarray
        The distance in Angstroms between each couple of amino acids in the
        peptide chain.
    norm_contact_map : np.ndarray
        The contact map in a scale between 0 and 1.
    binary_contact_map : np.ndarray
        The contact map binarized using two thresholding criteria.
    attention : tuple[torch.Tensor, ...]
        The attention from the model, cleared of the attention relative to
        tokens [CLS] and [SEP].
    attention_avgs : list[torch.Tensor]
        The averages of the attention masks independently computed for
        each layer and, as last element, the average of those averages.
    attention_to_amino_acids : torch.Tensor
        The attention given to each amino acid by each attention head.
    attention_sim_df : pd.DataFrame
        The attention similarity between each couple of amino acids.
    head_att_align : np.ndarray
        Array with shape (number_of_layers, number_of_heads), storing how much
        attention aligns with indicator_function for each attention masks.
    layer_att_align : np.ndarray
        Array with shape (number_of_layers), storing how much attention aligns
        with indicator_function for each average attention mask computed
        independently over each layer.
    seq_dir : Path
        The path to the folder containing the plots relative to the peptide
        chain.
    chain_amino_acids : list[str]
        The single letter codes of the amino acid types in the peptide chain.

    Returns
    -------
    None

    """
    config = config_parser.Config("config.txt")

    cutoffs = config.get_cutoffs()
    distance_cutoff = cutoffs["DISTANCE_CUTOFF"]
    position_cutoff = cutoffs["POSITION_CUTOFF"]

    seq_ID = seq_dir.stem

    # 1.1-1.2
    with Loading("Plotting distance and contact maps"):
        plot_distance_and_contact(distance_map, norm_contact_map, seq_dir)
    # 1.3
    with Loading("Plotting binary contact map"):
        plot_path = seq_dir/f"{seq_ID}_binary_contact_map.png"
        if plot_path.is_file() is False:
            fig, ax = plt.subplots()
            ax.set_title(
                f"{seq_ID}\nBinary Contact Map - Cutoff {distance_cutoff} Å\n"
                f"Excluding Contacts within {position_cutoff} Positions"
            )
            ax.imshow(binary_contact_map, cmap='Blues')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
    # 1.4
    with Loading("Plotting attention matrices"):
        plot_attention_matrices(
            attention,
            plot_title="{seq_ID}\nAttention Matrices - "
            "Layer {layer_number}".format(seq_ID=seq_ID, layer_number=30)
        )
    # 1.5
    with Loading("Plotting attention averages per layer"):
        plot_attention_matrices(
            tuple(attention_avgs[:-1]),
            plot_title=f"{seq_ID}\nAverages of the Attention per Layer"
        )
    # 1.6
    with Loading("Plotting attention average over the whole model"):
        plot_attention_matrices(
            attention_avgs[-1],
            plot_title=f"{seq_ID}\nAverage of the Attention over the whole "
            "Model"
        )
    # 1.7
    with Loading("Plotting attention to amino acids"):
        plot_attention_to_amino_acids_together(
            attention_to_amino_acids, chain_amino_acids,
            plot_title=f"{seq_ID}\nAttention to Amino Acids"
        )
    # 1.8
    with Loading("Plotting attention similarity"):
        plot_heatmap(
            attention_sim_df,
            plot_title=f"{seq_ID}\nPairwise Attention Similarity - "
            "Pearson Correlation"
        )
    # 1.9
    with Loading("Plotting attention alignment"):
        plot_heatmap(
            head_attention_align,
            plot_title=f"{seq_ID}\nAttention Alignment"
        )
    # 1.10
    with Loading("Plotting attention alignment per layer"):
        plot_bars(
            layer_attention_align,
            plot_title=f"{seq_ID}\nAttention Alignment per Layer"
        )

    plt.close('all')


def plot_on_set(
    avg_PT_att_to_amino_acids: torch.Tensor,
    avg_PWT_att_to_amino_acids: torch.Tensor,
    avg_PH_att_to_amino_acids: torch.Tensor,
    avg_att_sim_arr: np.ndarray,
    avg_head_att_align: np.ndarray,
    avg_layer_att_align: np.ndarray,
    sum_amino_acid_df: pd.DataFrame,
) -> None:
    """
    Plot and save the arguments received.

    Parameters
    ----------
    avg_PT_att_to_amino_acids : torch.Tensor
        The percentage of total attention given to each amino acid, averaged
        over the whole protein set.
    avg_PWT_att_to_amino_acids : torch.Tensor
        The percentage of total attention given to each amino acid, averaged
        over the whole protein set and weighted by the occurrences of that
        amino acid along all the proteins.
    avg_PH_att_to_amino_acids : torch.Tensor
        The percentage of each head's attention given to each amino acid,
        averaged over the whole protein set.
    avg_att_sim_arr : np.ndarray
        The attention similarity averaged over the whole protein set.
    avg_head_att_align : np.ndarray
        The head attention alignment averaged over the whole protein set.
    avg_layer_att_align : np.ndarray
        The layer attention alignment averaged over the whole protein set.
    sum_amino_acid_df : pd.DataFrame
        The data frame containing the information about all the amino acids
        in the set of proteins.
    Returns
    -------
    None

    """
    # 2.1
    with Loading(
        "Plotting average percentage of total attention to amino acids"
    ):
        plot_attention_to_amino_acids_together(
            avg_PT_att_to_amino_acids,
            sum_amino_acid_df["Amino Acid"].to_list(),
            plot_title="Average Percentage of Total Attention to each Amino "
            "Acid"
        )
    # 2.2
    with Loading(
        "Plotting average percentage of weighted total attention to amino "
        "acids"
    ):
        plot_attention_to_amino_acids_together(
            avg_PWT_att_to_amino_acids,
            sum_amino_acid_df["Amino Acid"].to_list(),
            plot_title="Average Percentage of Weighted Total Attention to each"
            " Amino Acid"
        )
    # 2.3
    with Loading(
        "Plotting average percentage of heads' attention to amino acids"
    ):
        plot_attention_to_amino_acids_alone(
            avg_PH_att_to_amino_acids,
            sum_amino_acid_df["Amino Acid"].to_list(),
            plot_title="Average Percentage of each Head's Attention to:"
        )
    # 2.4
    with Loading("Plotting average attention similarity"):
        plot_heatmap(
            avg_att_sim_arr, plot_title="Average Pairwise Attention Similarity"
            "\nPearson Correlation"
        )
    # 2.5
    with Loading("Plotting average head attention alignment"):
        plot_heatmap(
            avg_head_att_align, plot_title="Average Head Attention Alignment"
        )
    # 2.6
    with Loading("Plotting average layer attention alignment"):
        plot_bars(
            avg_layer_att_align, plot_title="Average Layer Attention Alignment"
        )

    plt.close('all')
