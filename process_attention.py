#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process attention.

This script computes:
    - the attention similarity between couples of amino acids
    - the averages of the attention masks independently computed for each
      layer, and the average of those averages, which refers to the whole model
    - the attention alignments for each attention masks, for the averages of
      each layer, and for the total average referring to the whole model
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'


from modules.attention import average_masks_together, \
    compute_attention_alignment, compute_attention_similarity

import numpy as np
import pandas as pd
import torch


def main(attention: tuple,
         attention_to_amino_acids: torch.Tensor,
         indicator_function: np.ndarray
         ) -> (pd.DataFrame, tuple, torch.Tensor, np.ndarray, np.ndarray,
               float):
    """
    Compute attention similarity, attention averages and attention alignments.

    Parameters
    ----------
    attention : tuple
        contains tensors that store the attention from the model, cleared of
        the attention relative to tokens [CLS] and [SEP]
    attention_to_amino_acids : torch.Tensor
        tensor having dimension (number_of_amino_acids, number_of_layers,
        number_of_heads), storing the absolute attention given to each amino
        acid by each attention head
    indicator_function : np.ndarray
        binary map representing one property of the peptide chain (returns 1 if
        the property is present, 0 otherwise)

    Returns
    -------
    attention_sim_df : pd.DataFrame
        stores attention similarity between each couple of amino acids
    attention_per_layer : tuple
        averages of the attention masks in each layer
    model_attention_average : torch.Tensor
        average of the average attention masks per layer
    head_attention_alignment : np.ndarray
        array having dimension (number_of_layers, number_of_heads), storing how
        much attention aligns with indicator_function for each attention masks
    layer_attention_alignment : np.ndarray
        array having dimension (number_of_layers), storing how much attention
        aligns with indicator_function for each average attention mask computed
        independently over each layer
    model_attention_alignment : float
        fraction of attention that aligns with indicator_function for the
        average attention mask of the model

    """
    attention_sim_df = compute_attention_similarity(attention_to_amino_acids)

    attention_per_layer, model_attention_average = average_masks_together(
        attention)

    head_attention_alignment = compute_attention_alignment(
        attention, indicator_function)
    layer_attention_alignment = compute_attention_alignment(
        attention_per_layer, indicator_function)
    model_attention_alignment = compute_attention_alignment(
        model_attention_average, indicator_function)

    return (attention_sim_df, attention_per_layer, model_attention_average,
            head_attention_alignment, layer_attention_alignment,
            model_attention_alignment)
