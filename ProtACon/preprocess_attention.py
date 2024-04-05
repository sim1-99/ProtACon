#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-process attention.

This pre-processes the attention returned by ProtBert by removing the attention
relative to the tokens [CLS] and [SEP]. Then returns the attention given by the
model to each amino acid.
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from pathlib import PosixPath

from ProtACon.modules.attention import (
    compute_weighted_attention,
    get_amino_acid_pos,
    get_attention_to_amino_acid,
    sum_attention_on_columns
    )

import pandas as pd
import torch


def main(
        attention: tuple[torch.Tensor, ...],
        tokens: list[str],
        seq_dir: PosixPath
         ) -> list[
             pd.DataFrame,
             torch.Tensor,
             torch.Tensor,
             torch.Tensor
             ]:
    """
    Pre-process attention from ProtBert.

    raw_attention and raw_tokens are cleared of the tokens [CLS] and [SEP],
    then attention given to each amino acid is computed. It also contructs a
    data frame containing information about the amino acids in the input
    peptide chain.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        contains tensors that store the attention from the model, cleared of
        the attention relative to tokens [CLS] and [SEP]
    tokens : list[str]
        contains strings which are the tokens used by the model, cleared of the
        tokens [CLS] and [SEP]
    seq_dir : PosixPath
        path to the folder containing the plots relative to the peptide chain

    Returns
    -------
    list
        amino_acid_df : pd.DataFrame
            contains information about the amino acids in the input peptide
            chain
        attention_to_amino_acids : torch.Tensor
            tensor having dimension (number_of_amino_acids, number_of_layers,
            number_of_heads), storing the absolute attention given to each
            amino acid by each attention head
        rel_attention_to_amino_acids : torch.Tensor
            tensor having dimension (number_of_amino_acids, number_of_layers,
            number_of_heads), storing the relative attention in percentage
            given to each amino acid by each attention head; "relative" means
            that the values of attention given by one head to one amino acid
            are divided by the total value of attention of that head
        weight_attention_to_amino_acids : torch.Tensor
            tensor resulting from weighting rel_attention_to_amino_acids by the
            number of occurrences of the corresponding amino acid

    """
    attention_on_columns = sum_attention_on_columns(attention)

    # remove duplicate amino acids from tokens and store the rest in a list
    types_of_amino_acids = list(dict.fromkeys(tokens))

    # create two empty lists
    attention_to_amino_acids = list(range(len(types_of_amino_acids)))
    rel_attention_to_amino_acids = list(range(len(types_of_amino_acids)))

    # start data frame construction
    columns = ["Amino Acid", "Occurrences", "Percentage Frequency (%)",
               "Position in Token List"]
    amino_acid_df = pd.DataFrame(
        data=None, index=range(len(types_of_amino_acids)), columns=columns)

    for amino_acid_idx, amino_acid in enumerate(types_of_amino_acids):
        amino_acid_df.at[amino_acid_idx, "Amino Acid"] = amino_acid

        amino_acid_df.at[amino_acid_idx, "Position in Token List"
                         ] = get_amino_acid_pos(amino_acid, tokens)

        amino_acid_df.at[amino_acid_idx, "Occurrences"] = len(
            amino_acid_df.at[amino_acid_idx, "Position in Token List"])

        amino_acid_df.at[
            amino_acid_idx, "Percentage Frequency (%)"
            ] = amino_acid_df.at[amino_acid_idx, "Occurrences"]/len(tokens)*100

    # sort the residue types by alphabetical order
    amino_acid_df.sort_values(by=["Amino Acid"], inplace=True)

    # take into account the previous sorting when calculate att to amino acids
    for list_idx, df_idx in zip(
            range(len(types_of_amino_acids)), amino_acid_df.index):
        attention_to_amino_acids[list_idx], rel_attention_to_amino_acids[
            list_idx] = get_attention_to_amino_acid(
                attention_on_columns,
                amino_acid_df.at[df_idx, "Position in Token List"])
    # end data frame construction

    seq_ID = seq_dir.stem
    amino_acid_df.to_csv(
        seq_dir/f"{seq_ID}_residue_df.csv", index=False, columns=columns,
        sep=';')

    attention_to_amino_acids = torch.stack(attention_to_amino_acids)

    rel_attention_to_amino_acids = torch.stack(rel_attention_to_amino_acids)

    weight_attention_to_amino_acids = compute_weighted_attention(
        rel_attention_to_amino_acids, amino_acid_df)

    return [amino_acid_df,
            attention_to_amino_acids,
            rel_attention_to_amino_acids,
            weight_attention_to_amino_acids
            ]
