"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-08-14

Compute and save the averages of:

- the percentage of attention given to each amino acid
- the percentage of attention given to each amino acid, weighted by the
occurrences of that amino acid in all the proteins of the set
- the attention similarity
- the attention-contact alignment in the attention heads
- the attention-contact alignment across the layers

"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ProtACon import config_parser
from ProtACon.modules.attention import compute_attention_similarity
from ProtACon.modules.utils import Loading


def main(
    sum_att_head_sum: torch.Tensor,
    sum_att_to_am_ac: torch.Tensor,
    sum_head_att_align: np.ndarray,
    sum_layer_att_align: np.ndarray,
    sum_amino_acid_df: pd.DataFrame,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute attention alignment and similarity over the whole set of proteins.

    Parameters
    ----------
    sum_att_head_sum : torch.Tensor
        Tensor with shape (number_of_layers, number_of_heads), resulting from
        the sum over the set of proteins of the sum over all the values of each
        attention matrix.
    sum_att_to_am_ac : torch.Tensor
        Tensor with shape (len(sum_amino_acid_df), number_of_layers,
        number_of_heads), storing the sum over the set of proteins of the
        absolute attention given to each amino acid by each attention head.
    sum_head_att_align : np.ndarray
        The sum over the set of proteins of the arrays, each one with shape
        (number_of_layers, number_of_heads), storing how much attention aligns
        with indicator_function for each attention matrix.
    sum_layer_att_align : np.ndarray
        The sum over the set of proteins of the arrays, each one with shape
        (number_of_layers), storing how much attention aligns with
        indicator_function for each average attention matrix computed
        independently over each layer.
    sum_amino_acid_df : pd.DataFrame
        The data frame containing the information about all the amino acids
        in the set of proteins.

    Returns
    -------
    avg_P_att_to_am_ac : torch.Tensor
        The percentage of attention given to each amino acid, averaged over the
        whole protein set.
    avg_PW_att_to_am_ac : torch.Tensor
        The percentage of attention given to each amino acid, averaged over the
        whole protein set and weighted by the occurrences of that amino acid
        along all the proteins.
    avg_att_sim_df : pd.DataFrame
        The attention similarity averaged over the whole protein set.
    avg_head_att_align : np.ndarray
        The head attention alignment averaged over the whole protein set.
    avg_layer_att_align : np.ndarray
        The layer attention alignment averaged over the whole protein set.

    """
    config = config_parser.Config("config.txt")
    paths = config.get_paths()

    file_folder = paths["FILE_FOLDER"]
    file_dir = Path(__file__).resolve().parents[1]/file_folder

    with Loading("Saving average percentage of attention to amino acids"):
        avg_P_att_to_am_ac = 100*torch.div(
            sum_att_to_am_ac,
            torch.sum(sum_att_to_am_ac),
        )
        torch.save(
            avg_P_att_to_am_ac, file_dir/"avg_P_att_to_amino_acids.pt"
        )

    with Loading(
        "Saving average percentage of weighted attention to amino acids"
    ):
        occurrences = torch.tensor(
            sum_amino_acid_df["Occurrences"].to_list()
        )
        avg_W_att_to_am_ac = torch.div(
            sum_att_to_am_ac, occurrences.unsqueeze(1).unsqueeze(1)
        )
        avg_PW_att_to_am_ac = 100*torch.div(
            avg_W_att_to_am_ac,
            torch.sum(avg_W_att_to_am_ac),
        )
        torch.save(
            avg_PW_att_to_am_ac, file_dir/"avg_PW_att_to_amino_acids.pt"
        )

    with Loading("Saving average attention similarity"):
        avg_att_sim_df = compute_attention_similarity(
            sum_att_to_am_ac, sum_amino_acid_df["Amino Acid"].to_list()
        )
        # set diagonal to NaN
        avg_att_sim_arr = avg_att_sim_df.to_numpy()
        np.fill_diagonal(avg_att_sim_arr, np.nan)
        avg_att_sim_df = pd.DataFrame(
            data=avg_att_sim_arr,
            index=avg_att_sim_df.index,
            columns=avg_att_sim_df.columns,
        )
        avg_att_sim_df.to_csv(
            file_dir/"attention_sim_df.csv", index=True, sep=';'
        )

    with Loading("Saving average head attention alignment"):
        avg_head_att_align = sum_head_att_align_list/number_of_samples
        np.save(file_dir/"avg_head_att_align.npy", avg_head_att_align)

    with Loading("Saving average layer attention alignment"):
        avg_layer_att_align = sum_layer_att_align_list/number_of_samples
        np.save(file_dir/"avg_layer_att_align.npy", avg_layer_att_align)

    return (
        avg_P_att_to_am_ac,
        avg_PW_att_to_am_ac,
        avg_att_sim_df,
        avg_head_att_align,
        avg_layer_att_align,
    )
