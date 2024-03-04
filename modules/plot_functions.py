#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot functions.

This module contains the plotting functions of ProtACon (attention masks,
attention heatmaps, contact maps, etc.).
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'


from modules.utils import dict_1_to_3, get_model_structure
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def find_best_nrows(number_of_amino_acid_types: int) -> int:
    """
    Find the adequate number of rows to use in plt.subplots.

    Parameters
    ----------
    number_of_amino_acid_types : int

    Raises
    ------
    ValueError
        if the types of amino acids in the chain are more than 20

    Returns
    -------
    nrows : int
        number of rows to be used in plt.subplots

    """
    ncols = 4
    quotient = number_of_amino_acid_types/ncols

    if quotient > 5:
        raise ValueError("Found more than 20 amino acids")

    if quotient > int(quotient):
        find_best_nrows.nrows = int(quotient)+1
    elif quotient == int(quotient):
        find_best_nrows.nrows = int(quotient)

    return find_best_nrows.nrows


def plot_attention_to_amino_acids(attention_to_amino_acids: torch.Tensor,
                                  plot_title: str, types_of_amino_acids: list):
    """
    Plot attention heatmaps.

    Seaborn heatmaps are filled with the values of attention given to to each
    amino acid by each attention head.

    Parameters
    ----------
    attention_to_amino_acids : torch.Tensor
        tensor with dimension (number_of_amino_acids, number_of_layers,
        number_of_heads), storing the attention given to each amino
        acid by each attention head
    plot_title : str
    types_of_amino_acids : list
        contains strings with single letter amino acid codes of the amino acid
        types in the peptide chain

    Raises
    ------
    ValueError
        if plt.subplots has got too many rows with respect to the number of
        types of the amino acids in the chain

    Returns
    -------
    None.

    """
    amino_acid_idx = 0
    ncols = 4
    nrows = find_best_nrows.nrows
    number_of_heads = get_model_structure.number_of_heads
    number_of_layers = get_model_structure.number_of_layers

    xticks = list(range(1, number_of_heads+1))
    xticks_labels = list(map(str, xticks))
    yticks = list(range(1, number_of_layers+1, 2))
    yticks_labels = list(map(str, yticks))

    empty_subplots = ncols*nrows-len(types_of_amino_acids)

    if empty_subplots < 0 or empty_subplots > 3:
        raise ValueError("Too many rows in plt.subplots")

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(20, 20), constrained_layout=True)
    fig.suptitle(plot_title, fontsize=18)
    while amino_acid_idx < len(types_of_amino_acids):
        for row in range(nrows):
            for col in range(ncols):
                img = attention_to_amino_acids[amino_acid_idx].detach().numpy()
                sns.heatmap(img, ax=axes[row, col])
                axes[row, col].set_title(
                    f"{dict_1_to_3[types_of_amino_acids[amino_acid_idx]][1]} "
                    f"({types_of_amino_acids[amino_acid_idx]})")
                axes[row, col].set_xlabel("Head")
                axes[row, col].set_xticks(xticks, labels=xticks_labels)
                axes[row, col].set_ylabel("Layer")
                axes[row, col].set_yticks(yticks, labels=yticks_labels)
                amino_acid_idx += 1

    for i in range(empty_subplots):
        fig.delaxes(axes[nrows, ncols-i])

    fig.savefig('TO_FIX')  # TODO: find a way to give unique titles
