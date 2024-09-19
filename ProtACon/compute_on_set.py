"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-08-14

Given a set of peptide chains, compute and save:

- the percentage of total attention given to each amino acid
- the percentage of total attention given to each amino acid, weighted by the
occurrences of that amino acid in all the proteins of the set
- the percentage of each head's attention given to each amino acid
- the global attention similarity between each couple of amino acids
- the attention-contact alignment in the attention heads, averaged over the
whole set of proteins
- the attention-contact alignment across the layers, averaged over the whole
set of proteins

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
    sum_att_to_aa: torch.Tensor,
    sum_head_att_align: np.ndarray,
    sum_layer_att_align: np.ndarray,
    sum_amino_acid_df: pd.DataFrame,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
]:
    """
    Compute the attention to the amino acids, the global attention similarity
    and the average attention alignment over the whole set of proteins.

    Parameters
    ----------
    sum_att_head_sum : torch.Tensor
        Tensor with shape (number_of_layers, number_of_heads), resulting from
        the sum over the set of proteins of the sum over all the values of each
        attention matrix.
    sum_att_to_aa : torch.Tensor
        Tensor with shape (len(sum_amino_acid_df), number_of_layers,
        number_of_heads), storing the sum over the set of proteins of the
        attention given to each amino acid by each attention head.
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
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        PT_att_to_aa : torch.Tensor
            The percentage of total attention given to each amino acid in the
            whole set of proteins.
        PWT_att_to_aa : torch.Tensor
            The percentage of total attention given to each amino acid in the
            whole set of proteins, weighted by the occurrences of that amino
            acid along all the proteins.
        PH_att_to_aa : torch.Tensor
            The percentage of each head's attention given to each amino acid,
            in the whole set of proteins.
    glob_att_sim_df : pd.DataFrame
        The attention similarity between each couple of amino acids in the
        whole set of proteins.
    tuple[np.ndarray, np.ndarray]
        avg_head_att_align : np.ndarray
            The head attention alignment averaged over the whole protein set.
        avg_layer_att_align : np.ndarray
            The layer attention alignment averaged over the whole protein set.

    """
    config = config_parser.Config("config.txt")
    paths = config.get_paths()
    proteins = config.get_proteins()

    file_folder = paths["FILE_FOLDER"]
    sample_size = proteins["SAMPLE_SIZE"]

    file_dir = Path(__file__).resolve().parents[1]/file_folder

    with Loading("Saving percentage of total attention to amino acids"):
        PT_att_to_aa = 100*torch.div(
            sum_att_to_aa,
            torch.sum(sum_att_to_aa),
        )
        torch.save(PT_att_to_aa, file_dir/"PT_att_to_aa.pt")

    with Loading(
        "Saving percentage of weighted total attention to amino acids"
    ):
        occurrences = torch.tensor(
            sum_amino_acid_df["Occurrences"].to_list()
        )
        WT_att_to_aa = torch.div(
            sum_att_to_aa, occurrences.unsqueeze(1).unsqueeze(1)
        )
        PWT_att_to_aa = 100*torch.div(
            WT_att_to_aa,
            torch.sum(WT_att_to_aa),
        )
        torch.save(PWT_att_to_aa, file_dir/"PWT_att_to_aa.pt")

    with Loading("Saving percentage of heads' attention to amino acids"):
        PH_att_to_aa = torch.div(sum_att_to_aa, sum_att_head_sum)*100
        # set to 0 the NaN values coming from the division by zero, in order to
        # improve the data visualization in the heatmaps
        PH_att_to_aa = torch.where(
            torch.isnan(PH_att_to_aa),
            torch.tensor(0, dtype=torch.float32),
            PH_att_to_aa,
        )
        torch.save(PH_att_to_aa, file_dir/"PH_att_to_aa.pt")

    with Loading("Saving attention similarity"):
        glob_att_sim_df = compute_attention_similarity(
            sum_att_to_aa, sum_amino_acid_df["Amino Acid"].to_list()
        )
        # set diagonal to NaN
        glob_att_sim_arr = glob_att_sim_df.to_numpy()
        np.fill_diagonal(glob_att_sim_arr, np.nan)
        glob_att_sim_df = pd.DataFrame(
            data=glob_att_sim_arr,
            index=glob_att_sim_df.index,
            columns=glob_att_sim_df.columns,
        )
        glob_att_sim_df.to_csv(file_dir/"att_sim_df.csv", index=True, sep=';')

    with Loading("Saving average head attention alignment"):
        avg_head_att_align = sum_head_att_align/sample_size
        np.save(file_dir/"avg_head_att_align.npy", avg_head_att_align)

    with Loading("Saving average layer attention alignment"):
        avg_layer_att_align = sum_layer_att_align/sample_size
        np.save(file_dir/"avg_layer_att_align.npy", avg_layer_att_align)

    return (
        (
            PT_att_to_aa,
            PWT_att_to_aa,
            PH_att_to_aa,
        ),
        glob_att_sim_df,
        (
            avg_head_att_align,
            avg_layer_att_align,
        ),
    )
