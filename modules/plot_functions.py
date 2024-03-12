#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot functions.

This module contains the plotting functions of ProtACon (attention masks,
attention heatmaps, contact maps, etc.).
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

import config_parser
from modules.utils import dict_1_to_3, get_types_of_amino_acids

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pathlib import Path, PosixPath
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


def plot_attention_masks(attention, plot_title: str):
    """
    Plot attention masks.

    Parameters
    ----------
    attention : torch.Tensor or tuple
    plot_title : str

    Returns
    -------
    None.

    """
    seq_ID = plot_title[0:4]

    config = config_parser.Config("config.txt")
    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    seq_dir = Path(__file__).parent.parent/plot_folder/seq_ID

    if type(attention) is torch.Tensor:
        nrows = 1
        ncols = 1
    elif len(attention) == 30:
        if len(attention[0].size()) == 2:
            nrows = 6
            ncols = 5
        elif len(attention[0].size()) == 3:
            nrows = 4
            ncols = 4
            layer_number = int(plot_title[-2:])
    attention_head_idx = 0

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    fig.suptitle(plot_title, fontsize=18)
    for row in range(nrows):
        for col in range(ncols):
            if type(attention) is torch.Tensor:
                img = attention.detach().numpy()
                axes.imshow(img, cmap='Blues')
            elif len(attention) == 30:
                if len(attention[0].size()) == 2:
                    img = attention[attention_head_idx].detach().numpy()
                    axes[row, col].set_title(f"Layer {attention_head_idx+1}")
                elif len(attention[0].size()) == 3:
                    img = attention[layer_number-1][attention_head_idx
                                                    ].detach().numpy()
                    axes[row, col].set_title(f"Head {attention_head_idx+1}")
                axes[row, col].set_xticks([])
                axes[row, col].imshow(img, cmap='Blues')
            attention_head_idx += 1

    if "Model" in plot_title:
        plt.savefig(seq_dir/f"{seq_ID}_att_mask_model_avg.png")
    elif "Layer" and "Averages" in plot_title:
        plt.savefig(seq_dir/f"{seq_ID}_att_masks_layer_avg.png")
    else:
        plt.savefig(seq_dir/f"{seq_ID}_att_masks_layer_{layer_number}.png")

    plt.close()


def plot_attention_to_amino_acids(attention_to_amino_acids: torch.Tensor,
                                  plot_title: str):
    """
    Plot attention heatmaps.

    Seaborn heatmaps are filled with the values of attention given to to each
    amino acid by each attention head.

    Parameters
    ----------
    attention_to_amino_acids : torch.Tensor
        tensor having dimension (number_of_amino_acids, number_of_layers,
        number_of_heads), storing the attention given to each amino acid by
        each attention head
    plot_title : str

    Raises
    ------
    ValueError
        if plt.subplots has got too many rows with respect to the number of
        types of the amino acids in the chain

    Returns
    -------
    None.

    """
    types_of_amino_acids = get_types_of_amino_acids.types_of_amino_acids
    seq_ID = plot_title[0:4]

    config = config_parser.Config("config.txt")
    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    seq_dir = Path(__file__).parent.parent/plot_folder/seq_ID

    amino_acid_idx = 0
    ncols = 4
    nrows = find_best_nrows.nrows

    xticks = list(range(1, attention_to_amino_acids.size(dim=2)+1))
    xticks_labels = list(map(str, xticks))
    yticks = list(range(1, attention_to_amino_acids.size(dim=1)+1, 2))
    yticks_labels = list(map(str, yticks))

    empty_subplots = ncols*nrows-len(types_of_amino_acids)

    if empty_subplots < 0 or empty_subplots > 3:
        raise ValueError("Too many rows in plt.subplots")

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(20, 20), constrained_layout=True)
    fig.suptitle(plot_title, fontsize=18)
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
            if amino_acid_idx < len(types_of_amino_acids)-1:
                amino_acid_idx += 1
            else:
                break

    for i in range(empty_subplots):
        fig.delaxes(axes[nrows-1, ncols-1-i])

    if "Percentage" in plot_title:
        fig.savefig(seq_dir/f"{seq_ID}_P_att_to_aa.png")
    elif "Weighted" in plot_title:
        fig.savefig(seq_dir/f"{seq_ID}_WP_att_to_aa.png")
    else:
        fig.savefig(seq_dir/f"{seq_ID}_att_to_aa.png")

    plt.close()


def plot_distance_and_contact(distance_map: np.ndarray,
                              norm_contact_map: np.ndarray,
                              seq_dir: PosixPath):
    """
    Plot the distance map and the normalized contact map side by side.

    Parameters
    ----------
    distance_map : np.ndarray
        it shows the distance - expressed in Angstroms - between each couple of
        amino acids in the peptide chain
    norm_contact_map : np.ndarray
        it shows how much each amino acid is close to all the others, in a
        scale between 0 and 1
    seq_dir : PosixPath
        path to the folder containing the plots relative to the peptide chain

    Returns
    -------
    None.

    """
    seq_ID = seq_dir.stem

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(121)
    ax1.title.set_text(f"{seq_ID}\nDistance Map")
    im1 = ax1.imshow(distance_map, cmap='Blues')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    ax2 = fig.add_subplot(122)
    ax2.title.set_text(f"{seq_ID}\nNormalized Contact Map")
    im2 = ax2.imshow(norm_contact_map, cmap='Blues')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    plt.savefig(seq_dir/f"{seq_ID}_distance_and_contact.png")
    plt.close()


def plot_heatmap(attention, plot_title: str):
    """
    Plot sns.heatmap.

    Parameters
    ----------
    attention : pd.DataFrame or np.ndarray
        any data structure having dimension (number_of_layers, number_of_heads)
    plot_title : str

    Returns
    -------
    None.

    """
    seq_ID = plot_title[0:4]

    config = config_parser.Config("config.txt")
    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    seq_dir = Path(__file__).parent.parent/plot_folder/seq_ID

    fig, ax = plt.subplots()
    sns.heatmap(attention)
    ax.set_title(plot_title)

    if type(attention) is np.ndarray:
        xticks = list(range(1, attention.shape[1]+1))
        xticks_labels = list(map(str, xticks))
        yticks = list(range(1, attention.shape[0]+1, 2))
        yticks_labels = list(map(str, yticks))

        ax.set_xlabel("Head")
        ax.set_xticks(xticks, labels=xticks_labels)
        ax.set_ylabel("Layer")
        ax.set_yticks(yticks, labels=yticks_labels)

    if "Alignment" in plot_title:
        plt.savefig(seq_dir/f"{seq_ID}_att_align_heads.png")
    elif "Similarity" in plot_title:
        plt.savefig(seq_dir/f"{seq_ID}_att_sim.png")

    plt.close()
