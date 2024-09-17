"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

This script computes:
    - the attention similarity between couples of types of amino acids
    - the averages of the attention matrices independently computed for each
      layer, and the average of those averages, which refers to the whole model
    - the attention alignments for each attention matrix, for the averages of
      each layer, and for the total average referring to the whole model

"""
import numpy as np
import pandas as pd
import torch

from ProtACon.modules.attention import (
    average_matrices_together,
    compute_attention_alignment,
    compute_attention_similarity,
)
from ProtACon.modules.miscellaneous import all_amino_acids


def main(
    attention: tuple[torch.Tensor, ...],
    att_to_amino_acids: torch.Tensor,
    indicator_function: np.ndarray,
    chain_amino_acids: list[str],
) -> tuple[
    pd.DataFrame,
    list[torch.Tensor],
    np.ndarray,
    np.ndarray,
]:
    """
    Compute attention similarity, attention averages and attention alignments.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention from the model, cleared of the attention relative to
        tokens [CLS] and [SEP].
    att_to_amino_acids : torch.Tensor
        Tensor with shape (len(all_amino_acids), number_of_layers,
        number_of_heads), storing the absolute attention given to each amino
        acid by each attention head.
    indicator_function : np.ndarray
        The binary map representing one property of the peptide chain (return
        1 if the property is present, 0 otherwise).
    chain_amino_acids : list[str]
        The single letter codes of the amino acid types in the peptide chain.

    Returns
    -------
    attention_sim_df : pd.DataFrame
        The attention similarity between each couple of amino acids.
    attention_avgs : list[torch.Tensor]
        The averages of the attention matrices independently computed for each
        layer and, as last element, the average of those averages.
    head_attention_alignment : np.ndarray
        Array with shape (number_of_layers, number_of_heads), storing how much
        attention aligns with indicator_function for each attention matrix.
    layer_attention_alignment : np.ndarray
        Array with shape (number_of_layers), storing how much attention aligns
        with indicator_function for each average attention matrix computed
        independently over each layer.

    """
    nonzero_indices = [
        all_amino_acids.index(type) for type in chain_amino_acids
    ]
    att_to_amino_acids = torch.index_select(
        att_to_amino_acids, 0, torch.tensor(nonzero_indices)
    )

    attention_sim_df = compute_attention_similarity(
        att_to_amino_acids, chain_amino_acids
    )

    attention_avgs = average_matrices_together(attention)

    head_attention_alignment = compute_attention_alignment(
        attention, indicator_function
    )
    layer_attention_alignment = compute_attention_alignment(
        tuple(attention_avgs[:-1]), indicator_function
    )

    return (
        attention_sim_df,
        attention_avgs,
        head_attention_alignment,
        layer_attention_alignment,
    )
