"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Compute:
    - the attention similarity between couples of amino acids
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


def main(
    attention: tuple[torch.Tensor, ...],
    att_to_aa: torch.Tensor,
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
    att_to_aa : torch.Tensor
        Tensor with shape (len(chain_amino_acids), n_layers, n_heads), storing
        the attention given to each amino acid by each attention head.
    indicator_function : np.ndarray
        The binary map representing one property of the peptide chain (return
        1 if the property is present, 0 otherwise).
    chain_amino_acids : list[str]
        The single letter codes of the amino acids in the peptide chain.

    Returns
    -------
    att_sim_df : pd.DataFrame
        The attention similarity between each couple of amino acids.
    att_avgs : list[torch.Tensor]
        The averages of the attention matrices independently computed for each
        layer and, as last element, the average of those averages.
    head_att_align : np.ndarray
        Array with shape (n_layers, n_heads), storing how much attention aligns
        with indicator_function for each attention matrix.
    layer_att_align : np.ndarray
        Array with shape (n_layers), storing how much attention aligns with
        indicator_function for each average attention matrix computed
        independently over each layer.

    """
    att_sim_df = compute_attention_similarity(
        att_to_aa, chain_amino_acids
    )

    att_avgs = average_matrices_together(attention)

    head_att_align = compute_attention_alignment(
        attention, indicator_function
    )
    layer_att_align = compute_attention_alignment(
        tuple(att_avgs[:-1]), indicator_function
    )

    return (
        att_sim_df,
        att_avgs,
        head_att_align,
        layer_att_align,
    )
