"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 14-08-2024

"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ProtACon import config_parser
from ProtACon.modules.utils import (
    average_arrs_together,
    average_dfs_together,
    Loading,
)


config = config_parser.Config("config.txt")

paths = config.get_paths()
plot_folder = paths["PLOT_FOLDER"]
plot_dir = Path(__file__).resolve().parents[1]/plot_folder


def main(
    rel_att_to_amino_acids: torch.Tensor,
    weight_att_to_amino_acids: torch.Tensor,
    att_sim_df_list: list[pd.DataFrame],
    head_att_align_list: list[np.ndarray],
    layer_att_align_list: list[np.ndarray],
    number_of_samples: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute attention alignment and similarity over the whole set of proteins.

    Parameters
    ----------
    rel_att_to_amino_acids : torch.Tensor
        Tensor with shape (number_of_amino_acids, number_of_layers,
        number_of_heads), storing the relative attention in percentage given to
        each amino acid by each attention head; "rel" (relative) means that the
        values of attention given by one head to one amino acid are divided by
        the total value of attention of that head.
    weight_att_to_amino_acids : torch.Tensor
        The tensor resulting from weighting rel_att_to_amino_acids by the
        number of occurrences of the corresponding amino acid.
    att_sim_df_list : list[pd.DataFrame]
        The attention similarity between each couple of amino acids for each
        peptide chain.
    head_att_align_list : list[np.ndarray]
        The arrays, one for each peptide chain, each one having dimension
        (number_of_layers, number_of_heads), storing how much attention aligns
        with indicator_function for each attention masks.
    layer_att_align_list : list[np.ndarray]
        The arrays, one for each peptide chain, each one having dimension
        (number_of_layers), storing how much attention aligns with
        indicator_function for each average attention mask computed
        independently over each layer.
    number_of_samples : int
        The number of proteins in the set.

    Returns
    -------
    avg_P_att_to_amino_acids : torch.Tensor
        The percentage of attention given to each amino acid by each attention
        head, averaged over the whole protein set.
    avg_PW_att_to_amino_acids : torch.Tensor
        The percentage of weighted attention given to each amino acid by each
        attention head, averaged over the whole protein set.
    avg_att_sim_df : pd.DataFrame
        The attention similarity averaged over the whole protein set.
    avg_head_att_align : np.ndarray
        The head attention alignment averaged over the whole protein set.
    avg_layer_att_align : np.ndarray
        The layer attention alignment averaged over the whole protein set.

    """
    with Loading("Computing average percentage of attention to amino acids"):
        avg_P_att_to_amino_acids = rel_att_to_amino_acids/number_of_samples*100

    with Loading(
        "Computing average percentage of weighted attention to amino acids"
    ):
        avg_PW_att_to_amino_acids = \
            weight_att_to_amino_acids/number_of_samples*100

    with Loading("Computing average attention similarity"):
        avg_att_sim_df = average_dfs_together(att_sim_df_list)

    avg_att_sim_df.to_csv(
        plot_dir/"attention_sim_df.csv", index=True, sep=';')

    with Loading("Computing average head attention alignment"):
        avg_head_att_align = average_arrs_together(head_att_align_list)

    with Loading("Computing average layer attention alignment"):
        avg_layer_att_align = average_arrs_together(layer_att_align_list)

    return (
        avg_P_att_to_amino_acids,
        avg_PW_att_to_amino_acids,
        avg_att_sim_df,
        avg_head_att_align,
        avg_layer_att_align,
    )
