#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention.

This module contains functions for the extraction and processing of attention
from the ProtBert model.
"""


__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from modules.miscellaneous import get_model_structure

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch


def average_masks_together(attention: tuple[torch.Tensor]
                           ) -> (tuple[torch.Tensor], torch.Tensor):
    """
    Compute average attention masks.

    First, it averages together the attention masks independently for each
    layer, which are stored in attention_per_layer. Then, it averages together
    the attention masks in attention_per_layer, and stores them in
    model_attention_average.

    Parameters
    ----------
    attention : tuple[torch.Tensor]
        contains tensors that store the attention from the model, cleared of
        the attention relative to tokens [CLS] and [SEP]

    Returns
    -------
    attention_per_layer : tuple[torch.Tensor]
        tuple of tensors representing the averages of the attention masks in
        each layer
    model_attention_average : torch.Tensor
        average of the average attention masks per layer

    """
    number_of_layers = get_model_structure.number_of_layers

    attention_per_layer = list(range(number_of_layers))
    for layer_idx, layer in enumerate(attention):
        attention_per_layer[layer_idx] = torch.sum(layer, dim=0
                                                   )/layer.size(dim=0)
    model_attention_average = torch.sum(
        torch.stack(attention_per_layer), dim=0)/number_of_layers
    attention_per_layer = tuple(attention_per_layer)

    return attention_per_layer, model_attention_average


def clean_attention(raw_attention: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
    """
    Remove from the attention the one relative to non-amino acid tokens.

    Parameters
    ----------
    raw_attention : tuple[torch.Tensor]
        contains tensors that store the attention from the model, including the
        attention relative to tokens [CLS] and [SEP]

    Returns
    -------
    attention: tuple[torch.Tensor]
        contains tensors that store the attention from the model, cleared of
        the attention relative to tokens [CLS] and [SEP]

    """
    attention = []
    for layer_idx in range(len(raw_attention)):
        list_of_heads = []
        for head_idx in range(len(raw_attention[layer_idx][0])):
            list_of_heads.append(
                raw_attention[layer_idx][0][head_idx][1:-1, 1:-1])
        attention.append(torch.stack(list_of_heads))
    attention = tuple(attention)

    return attention


def compute_attention_alignment(attention: [torch.Tensor | tuple],
                                indicator_function: np.ndarray
                                ) -> [float | np.ndarray]:
    """
    Compute the proportion of attention that aligns with a certain property.

    The property is represented with the binary map indicator_function.

    Parameters
    ----------
    attention : torch.Tensor | tuple
    indicator_function : np.ndarray
        binary map representing one property of the peptide chain (returns 1 if
        the property is present, 0 otherwise)

    Returns
    -------
    attention_alignment : float |np.ndarray
        stores how much attention aligns with indicator_function

    """
    if type(attention) is torch.Tensor:
        attention = attention.numpy()
        attention_alignment = float(
            np.sum(attention*indicator_function)/np.sum(attention))

    if type(attention) is tuple:
        number_of_layers = get_model_structure.number_of_layers
        if len(attention[0].size()) == 2:
            attention_alignment = np.empty((number_of_layers))
            for layer_idx, layer in enumerate(attention):
                layer = layer.numpy()
                attention_alignment[layer_idx
                                    ] = np.sum(layer*indicator_function
                                               )/np.sum(layer)

        if len(attention[0].size()) == 3:
            number_of_heads = get_model_structure.number_of_heads
            attention_alignment = np.empty((number_of_layers, number_of_heads))
            for layer_idx, layer in enumerate(attention):
                for head_idx, head in enumerate(layer):
                    head = head.numpy()
                    attention_alignment[layer_idx, head_idx
                                        ] = np.sum(head*indicator_function
                                                   )/np.sum(head)
    return attention_alignment


def compute_attention_similarity(attention_to_amino_acids: torch.Tensor,
                                 types_of_amino_acids: list[str],
                                 ) -> pd.DataFrame:
    """
    Compute attention similarity.

    It assesses the similarity of the attention received by each amino acids
    for each couple of amio acids. This is achieved by computing the Pearson
    correlation between the proportion of attention that each amino acid
    receives across heads. The diagonal obviously returns a perfect
    correlation (because the attention similarity between one amino acid and
    itself is total). Therefore it is set to 0.

    Parameters
    ----------
    attention_to_amino_acids : torch.Tensor
        tensor having dimension (number_of_layers, number_of_heads) containing
        the attention (either absolute or percentage or weighted) given to each
        amino acid by each attention head
    types_of_amino_acids : list[str]
        contains strings with single letter amino acid codes of the amino acid
        types in the peptide chain

    Returns
    -------
    attention_sim_df : pd.DataFrame
        stores attention similarity between each couple of amino acids

    """
    number_of_heads = get_model_structure.number_of_heads
    number_of_layers = get_model_structure.number_of_layers

    attention_sim_df = pd.DataFrame(
        data=None, index=types_of_amino_acids, columns=types_of_amino_acids)
    attention_sim_df = attention_sim_df[attention_sim_df.columns].astype(float)

    for matrix1_idx, matrix1 in enumerate(attention_to_amino_acids):
        matrix1 = matrix1.numpy().reshape(
            (number_of_heads*number_of_layers, ))
        for matrix2_idx, matrix2 in enumerate(attention_to_amino_acids):
            matrix2 = matrix2.numpy().reshape(
                (number_of_heads*number_of_layers, ))
            corr = pearsonr(matrix1, matrix2)[0]
            attention_sim_df.at[types_of_amino_acids[matrix1_idx],
                                types_of_amino_acids[matrix2_idx]] = corr
            if matrix1_idx == matrix2_idx:
                attention_sim_df.at[types_of_amino_acids[matrix1_idx],
                                    types_of_amino_acids[matrix2_idx]] = np.nan

    return attention_sim_df


def compute_weighted_attention(percent_attention_to_amino_acids: torch.Tensor,
                               amino_acid_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute weighted attention given to each amino acid in the peptide chain.

    percent_attention_to_amino_acids is weighted on the number of occurrencies
    of each amino acid.

    Parameters
    ----------
    percent_attention_to_amino_acids : torch.Tensor
        tensor having dimension (number_of_amino_acids, number_of_layers,
        number_of_heads), storing the relative attention in percentage given to
        each amino acid by each attention head; "relative" means that the
        values of attention given by one head to one amino acid are divided by
        the total value of attention of that head
    amino_acid_df : pd.DataFrame
        contains information about the amino acids in the input peptide chain

    Returns
    -------
    weighted_attention_to_amino_acids : torch.Tensor
        tensor resulting from weighting percent_attention_to_amino_acids by the
        number of occurrences of the corresponding amino acid

    """
    weighted_attention_to_amino_acids = []
    occurrences = amino_acid_df["Occurrences"].tolist()

    for percent_attention_to_amino_acid, occurrence in zip(
            percent_attention_to_amino_acids, occurrences):
        weighted_attention_to_amino_acids.append(
            percent_attention_to_amino_acid/occurrence)

    weighted_attention_to_amino_acids = torch.stack(
        weighted_attention_to_amino_acids)

    return weighted_attention_to_amino_acids


def get_amino_acid_pos(amino_acid: str, tokens: list[str]) -> list[str]:
    """
    Return the positions of a given token along the list of tokens.

    Parameters
    ----------
    amino_acid : str
        single letter amino acid code
    tokens : list[str]
        complete list of amino acid tokens

    Returns
    -------
    amino_acid_pos : list[str]
        positions of the tokens corresponding to amino_acid along the list
        tokens

    """
    amino_acid_pos = [idx for idx, token in enumerate(tokens)
                      if token == amino_acid]

    return amino_acid_pos


def get_attention_to_amino_acid(attention_on_columns: list[torch.Tensor],
                                amino_acid_pos: list[str]
                                ) -> (torch.Tensor, torch.Tensor):
    """
    Compute attention given from each attention head to each amino acid.

    The first tensor contains the absolute values of attention, while the
    second one contains the relative values in percentage. They take into
    account the fact that we do not consider attention to tokens [CLS] and
    [SEP]. If they are included, the sum of the attention values in each mask
    is the same in every head. This is no longer correct if the attention
    relative to those tokens is removed. Therefore, we have to correct possible
    distorsions that may rise as a consequence of that.

    Parameters
    ----------
    attention_on_columns : list[torch.Tensor]
        sum along each column of each attention mask; the sum along a column
        represent the attention given to the amino acid relative to the column
    amino_acid_pos : list[str]
        positions of the tokens corresponding to one amino acid along the list
        of tokens

    Returns
    -------
    attention_to_amino_acid : torch.Tensor
        tensor with dimension (number_of_layers, number_of_heads) containing
        the absolute attention given to each amino acid by each attention head
    percent_attention_to_amino_acid : torch.Tensor
        tensor with dimension (number_of_layers, number_of_heads) containing
        the relative attention in percentage given to each amino acid by each
        attention head; "relative" means that the values of attention given by
        one head to one amino acid are divided by the total value of attention
        of that head

    """
    number_of_heads = get_model_structure.number_of_heads
    number_of_layers = get_model_structure.number_of_layers

    # create two empty lists
    attention_to_amino_acid = list(range(len(attention_on_columns)))
    percent_attention_to_amino_acid = list(range(len(attention_on_columns)))

    """ collect the values of attention given to one amino acid by each head,
    then do the same with the next amino acid
    """
    for head_idx, head in enumerate(attention_on_columns):
        attention_to_amino_acid[head_idx] = head[amino_acid_pos[0]]
        for amino_acid_idx in range(1, len(amino_acid_pos)):
            """ since in each mask more than one column refer to the same amino
            acid, here we sum together all the "columns of attention" relative
            to the same amino acid
            """
            attention_to_amino_acid[head_idx] = torch.add(
                attention_to_amino_acid[head_idx],
                head[amino_acid_pos[amino_acid_idx]])

        """ here we compute the total value of attention of each mask, then
        we divide each value in attention_to_amino_acid by it and multiply by
        100 to express the values in percentage
        """
        sum_over_head = torch.sum(head)
        percent_attention_to_amino_acid[head_idx] = attention_to_amino_acid[
            head_idx]/sum_over_head*100

    attention_to_amino_acid = torch.stack(attention_to_amino_acid)
    attention_to_amino_acid = torch.reshape(
        attention_to_amino_acid, (number_of_layers, number_of_heads))

    percent_attention_to_amino_acid = torch.stack(
        percent_attention_to_amino_acid)
    percent_attention_to_amino_acid = torch.reshape(
        percent_attention_to_amino_acid, (number_of_layers, number_of_heads))

    return (attention_to_amino_acid, percent_attention_to_amino_acid)


def sum_attention(attention: tuple) -> list:  # TODO: delete if not used
    """
    Sum all values of attention of each attention mask in a tuple of tensors.

    Parameters
    ----------
    attention : tuple
        contains tensors that carry the attention returned by the model

    Returns
    -------
    total_head_attention : list
        contains (number_of_layers*number_of_heads) values resulting from the
        sum over all attention values of each attention mask

    """
    number_of_heads = get_model_structure.number_of_heads
    number_of_layers = get_model_structure.number_of_layers
    total_head_attention = list(range(number_of_layers*number_of_heads))

    for layer_idx, layer in enumerate(attention):
        for head_idx, head in enumerate(layer):
            total_head_attention[
                head_idx + layer_idx*number_of_heads] = float(torch.sum(head))

    return total_head_attention


def sum_attention_on_columns(attention: tuple[torch.Tensor]
                             ) -> list[torch.Tensor]:
    """
    Sum column-wise the values of attention of each mask in a tuple of tensors.

    Parameters
    ----------
    attention : tuple[torch.Tensor]
        contains tensors that store the attention returned by the model

    Returns
    -------
    attention_on_columns : list[torch.Tensor]
        contains (number_of_layers*number_of_heads) tensors, each with a length
        equal to the number of tokens, resulting from the column-wise sum over
        the attention values of each attention mask

    """
    number_of_heads = get_model_structure.number_of_heads
    number_of_layers = get_model_structure.number_of_layers
    attention_on_columns = list(range(number_of_layers*number_of_heads))

    for layer_idx, layer in enumerate(attention):
        for head_idx, head in enumerate(layer):
            attention_on_columns[
                head_idx + layer_idx*number_of_heads] = torch.sum(head, 0)

    return attention_on_columns
