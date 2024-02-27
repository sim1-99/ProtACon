#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process attention.

This script process the attention returned by ProtBert, and pre-processes it
by removing the attention relative to the tokens [CLS] and [SEP]. Then computes
the attention given by the model to each amino acid.
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'


from modules.attention import clean_attention, get_amino_acid_pos, \
    get_attention_to_amino_acid, sum_attention_on_columns
from modules.utils import get_model_structure
import pandas as pd
import torch


def main(raw_attention: tuple, raw_tokens: list) -> (
        pd.DataFrame, list, list, list):
    """
    Process attention from ProtBert.

    raw_attention and raw_tokens are cleared of the tokens [CLS] and [SEP],
    then attention given to each amino acid is computed. It also contructs a
    data frame containing information about the amino acids in the input
    peptide chain.

    Parameters
    ----------
    raw_attention : tuple
        contains tensors that store the attention from the model, including the
        attention relative to tokens [CLS] and [SEP]
    raw_tokens : list
        contains strings which are the tokens used by the model, including the
        tokens [CLS] and [SEP]

    Returns
    -------
    amino_acid_df : pd.DataFrame
        contains information about the amino acids in the input peptide chain
    attention_to_amino_acids : list
        contains tensors with dimension (number_of_layers, number_of_heads),
        storing the absolute attention given to each amino acid by each
        attention head
    percent_attention_to_amino_acids : list
        contains tensors with dimension (number_of_layers, number_of_heads),
        storing the relative attention in percentage given to each amino acid
        by each attention head; "relative" means that the values of attention
        given by one head to one amino acid are divided by the total value of
        attention of that head
    weighted_attention_to_amino_acids : list
        contains tensors resulting from weighting each tensor in
        percent_attention_to_amino_acids, by the number of occurrencies of the
        corresponding amino acid

    """
    number_of_heads, number_of_layers = get_model_structure(raw_attention)

    attention = clean_attention(raw_attention)
    tokens = raw_tokens[1:-1]

    attention_on_columns = sum_attention_on_columns(attention)

    # remove duplicate amino acids from tokens and store the rest in a list
    type_of_amino_acids = list(dict.fromkeys(tokens))

    # create two empty lists
    attention_to_amino_acids = list(range(len(type_of_amino_acids)))
    percent_attention_to_amino_acids = list(range(len(type_of_amino_acids)))

    # start data frame construction

    amino_acid_df = pd.DataFrame(
        data=None, index=range(len(type_of_amino_acids)), columns=[
            "Amino Acid", "Occurrences", "Percentage Frequency (%)",
            "Position in Token List"])

    for amino_acid_idx, amino_acid in enumerate(type_of_amino_acids):
        amino_acid_df.at[amino_acid_idx, "Amino Acid"] = amino_acid

        amino_acid_df.at[amino_acid_idx, "Position in Token List"
                         ] = get_amino_acid_pos(amino_acid, tokens)

        amino_acid_df.at[
            amino_acid_idx, "Occurrences"] = len(
                amino_acid_df.at[amino_acid_idx, "Position in Token List"])

        amino_acid_df.at[
            amino_acid_idx, "Percentage Frequency (%)"
            ] = amino_acid_df.at[amino_acid_idx, "Occurrences"]/len(tokens)*100

        attention_to_amino_acids[amino_acid_idx], \
            percent_attention_to_amino_acids[amino_acid_idx
                                             ] = get_attention_to_amino_acid(
            attention_on_columns, amino_acid_df.at[amino_acid_idx,
                                                   "Position in Token List"])

    # end data frame construction

    attention_to_amino_acids = torch.stack(attention_to_amino_acids)

    percent_attention_to_amino_acids = torch.stack(
        percent_attention_to_amino_acids)

    """ percent_attention_to_amino_acids weighted on the number of occurrencies
    of each amino acid
    """
    weighted_attention_to_amino_acids = []
    occurrencies = amino_acid_df["Occurrences"].tolist()

    for percent_attention_to_amino_acid, occurrency in zip(
            percent_attention_to_amino_acids, occurrencies):
        weighted_attention_to_amino_acids.append(
            percent_attention_to_amino_acid/occurrency)

    weighted_attention_to_amino_acids = torch.stack(
        weighted_attention_to_amino_acids)

    return (amino_acid_df,
            attention_to_amino_acids,
            percent_attention_to_amino_acids,
            weighted_attention_to_amino_acids)
