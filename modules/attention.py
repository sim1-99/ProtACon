#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention.

This module contains functions for the extraction and processing of attention
from the ProtBert model.
"""


__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from modules.utils import get_model_structure
import torch


def clean_attention(raw_attention: tuple) -> tuple:
    """
    Remove from the attention the one relative to non-amino acid tokens.

    Parameters
    ----------
    raw_attention : tuple
        contains tensors that store the attention from the model, including the
        attention relative to tokens [CLS] and [SEP]

    Returns
    -------
    tuple
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
    del list_of_heads

    return tuple(attention)


def get_amino_acid_pos(amino_acid: str, tokens: list) -> list:
    """
    Return the positions of a given token along the list of tokens.

    Parameters
    ----------
    amino_acid : str
        single letter amino acid code
    tokens : list
        complete list of amino acid tokens

    Returns
    -------
    amino_acid_pos : list
        positions of the tokens corresponding to amino_acid along tokens list

    """
    amino_acid_pos = [idx for idx, token in enumerate(tokens)
                      if token == amino_acid]

    return amino_acid_pos


def get_attention_to_amino_acid(
        attention_on_columns: list, amino_acid_pos: list) -> (torch.Tensor,
                                                              torch.Tensor):
    """
    Compute attention given from each attention head to each amino acid.

    The first tensor contain the absolute values of attention, while the second
    one contains the relative values in percentage. They take into account the
    fact that we do not consider attention to tokens [CLS] and [SEP]. If they
    are included, the sum of the attention values in each mask is the same in
    every head. This is no longer correct if we remove the attention relative
    to those tokens. Therefore, we have to correct possible distorsions that
    may rise as a consequence of that.

    Parameters
    ----------
    attention_on_columns : list
        sum along each column of each attention mask; the sum along a column
        represent the attention given to the amino acid relative to the column
    amino_acid_pos : list
        positions of the tokens corresponding to one amino acid along the list
        of tokens

    Returns
    -------
    tensor_attention_to_amino_acid : torch.Tensor
        tensor with dimension (number_of_layers, number_of_heads) containing
        the absolute attention given to each amino acid by each attention head
    tensor_percent_attention_to_amino_acid : torch.Tensor
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
    then make the same with the next amino acid
    """
    for head_idx, head in enumerate(attention_on_columns):
        attention_to_amino_acid[head_idx] = head[amino_acid_pos[0]]
        for amino_acid_idx in range(1, len(amino_acid_pos)):
            """ since in each mask more than one column refer to the same amino
            acid, here we sum together the all "columns of attention" relative
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

    tensor_attention_to_amino_acid = torch.stack(attention_to_amino_acid)
    tensor_attention_to_amino_acid = torch.reshape(
        tensor_attention_to_amino_acid, (number_of_layers, number_of_heads))

    tensor_percent_attention_to_amino_acid = torch.stack(
        percent_attention_to_amino_acid)
    tensor_percent_attention_to_amino_acid = torch.reshape(
        tensor_percent_attention_to_amino_acid, (number_of_layers,
                                                 number_of_heads))

    return (tensor_attention_to_amino_acid,
            tensor_percent_attention_to_amino_acid)


def sum_attention(attention: tuple) -> list:
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
    # number_of_heads = attention[0].shape[0]
    # number_of_layers = len(attention)
    number_of_heads = get_model_structure.number_of_heads
    number_of_layers = get_model_structure.number_of_layers
    total_head_attention = list(range(number_of_layers*number_of_heads))

    for layer_idx, layer in enumerate(attention):
        for head_idx, head in enumerate(layer):
            total_head_attention[
                head_idx + layer_idx*number_of_heads] = float(torch.sum(head))

    return total_head_attention


def sum_attention_on_columns(attention: tuple) -> list:
    """
    Sum column-wise the values of attention of each mask in a tuple of tensors.

    Parameters
    ----------
    attention : tuple
        contains tensors that carry the attention returned by the model

    Returns
    -------
    attention_on_columns : list
        contains (number_of_layers*number_of_heads) tensors, each with a length
        equal to the number of tokens, resulting from the column-wise sum over
        the attention values of each attention mask

    """
    # number_of_heads = attention[0].shape[0]
    # number_of_layers = len(attention)
    number_of_heads = get_model_structure.number_of_heads
    number_of_layers = get_model_structure.number_of_layers
    attention_on_columns = list(range(number_of_layers*number_of_heads))

    for layer_idx, layer in enumerate(attention):
        for head_idx, head in enumerate(layer):
            attention_on_columns[
                head_idx + layer_idx*number_of_heads] = torch.sum(head, 0)

    return attention_on_columns
