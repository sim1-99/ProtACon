"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Define the plot functions of ProtACon (attention masks, attention heatmaps,
contact maps, etc.).

"""
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
import torch

from ProtACon import config_parser
from ProtACon.modules.miscellaneous import dict_1_to_3
from ProtACon.modules.utils import Logger


log = Logger("cheesecake").get_logger()


def find_best_nrows(
    number_of_amino_acid_types: int,
) -> int:
    """
    Find the adequate number of rows to use in plt.subplots.

    Parameters
    ----------
    number_of_amino_acid_types : int

    Raises
    ------
    ValueError
        If the types of amino acids in the chain are more than 20.

    Returns
    -------
    nrows : int
        The number of rows to use in plt.subplots.

    """
    ncols = 4
    quotient = number_of_amino_acid_types/ncols

    if quotient > 5:
        raise ValueError("Found more than 20 amino acids")

    if quotient > int(quotient):
        nrows = int(quotient)+1
    elif quotient == int(quotient):
        nrows = int(quotient)

    return nrows


def plot_attention_masks(
    attention: torch.Tensor | tuple,
    plot_title: str,
) -> None:
    """
    Plot the attention masks.

    Parameters
    ----------
    attention : torch.Tensor | tuple
    plot_title : str

    Returns
    -------
    None

    """
    seq_ID = plot_title[0:4]

    config = config_parser.Config("config.txt")
    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    seq_dir = Path(__file__).resolve().parents[2]/plot_folder/seq_ID

    if type(attention) is torch.Tensor:
        nrows = 1
        ncols = 1
        plot_path = seq_dir/f"{seq_ID}_att_mask_model_avg.png"
    elif len(attention) == 30:
        if len(attention[0].size()) == 2:
            nrows = 6
            ncols = 5
            plot_path = seq_dir/f"{seq_ID}_att_masks_layer_avg.png"
        elif len(attention[0].size()) == 3:
            nrows = 4
            ncols = 4
            layer_number = int(plot_title[-2:])
            plot_path = seq_dir/f"{seq_ID}_att_masks_layer_{layer_number}.png"

    if plot_path.is_file():
        return None

    attention_head_idx = 0

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    fig.suptitle(plot_title, fontsize=18)
    for row in range(nrows):
        for col in range(ncols):
            if type(attention) is torch.Tensor:
                img = attention.numpy()
                plt.imshow(img, cmap='Blues')
            elif len(attention) == 30:
                if len(attention[0].size()) == 2:
                    img = attention[attention_head_idx].numpy()
                    axes[row, col].set_title(f"Layer {attention_head_idx+1}")
                elif len(attention[0].size()) == 3:
                    img = attention[layer_number-1][attention_head_idx].numpy()
                    axes[row, col].set_title(f"Head {attention_head_idx+1}")
                axes[row, col].set_xticks([])
                axes[row, col].imshow(img, cmap='Blues')
            attention_head_idx += 1

    plt.savefig(plot_path)
    plt.close()


def plot_attention_to_amino_acids(
    attention_to_amino_acids: torch.Tensor,
    amino_acids: list[str],
    plot_title: str,
) -> None:
    """
    Plot the attention heatmaps. The heatmaps are filled with the values of
    attention given to to each amino acid by each attention head.

    Parameters
    ----------
    attention_to_amino_acids : torch.Tensor
        Tensor with shape (number_of_amino_acids, number_of_layers,
        number_of_heads), storing the attention given to each amino acid by
        each attention head.
    amino_acids : list[str]
        The single letter codes of the amino acid types in the peptide chain or
        in the set of peptide chains.
    plot_title : str

    Raises
    ------
    ValueError
        If plt.subplots has got too many rows with respect to the number of
        types of the amino acids in the chain.

    Returns
    -------
    None

    """
    seq_ID = plot_title[0:4]

    config = config_parser.Config("config.txt")

    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    plot_dir = Path(__file__).resolve().parents[2]/plot_folder
    seq_dir = plot_dir/seq_ID

    if "Weighted" in plot_title:
        plot_path = plot_dir/"avg_PW_att_to_aa.png"
    elif "Percentage" in plot_title:
        plot_path = plot_dir/"avg_P_att_to_aa.png"
    else:
        plot_path = seq_dir/f"{seq_ID}_att_to_aa.png"

    if plot_path.is_file():
        log.logger.warning(
            f"A file with the same path already exists: {plot_path}\n"
            "The plot will not be saved."
        )
        return None

    amino_acid_idx = 0
    ncols = 4
    nrows = find_best_nrows(len(amino_acids))

    xticks = list(range(1, attention_to_amino_acids.size(dim=2)+1))
    xticks_labels = list(map(str, xticks))
    yticks = list(range(1, attention_to_amino_acids.size(dim=1)+1, 2))
    yticks_labels = list(map(str, yticks))

    empty_subplots = ncols*nrows-len(amino_acids)

    if empty_subplots < 0 or empty_subplots > 3:
        raise ValueError("Too many rows in plt.subplots")

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(20, 20), constrained_layout=True)
    fig.suptitle(plot_title, fontsize=18)
    for row in range(nrows):
        for col in range(ncols):
            img = attention_to_amino_acids[amino_acid_idx].numpy()
            sns.heatmap(img, ax=axes[row, col])
            axes[row, col].set_title(
                f"{dict_1_to_3[amino_acids[amino_acid_idx]][1]} "
                f"({amino_acids[amino_acid_idx]})"
            )
            axes[row, col].set_xlabel("Head")
            axes[row, col].set_xticks(xticks, labels=xticks_labels)
            axes[row, col].set_ylabel("Layer")
            axes[row, col].set_yticks(yticks, labels=yticks_labels)
            if amino_acid_idx < len(amino_acids)-1:
                amino_acid_idx += 1
            else:
                break

    for i in range(empty_subplots):
        fig.delaxes(axes[nrows-1, ncols-1-i])

    plt.savefig(plot_path)
    plt.close()


def plot_bars(
    attention: np.ndarray,
    plot_title: str,
) -> None:
    """
    Plot a barplot.

    Parameters
    ----------
    attention : np.ndarray
        Any data structure with shape (number_of_layers).
    plot_title : str

    Returns
    -------
    None

    """
    seq_ID = plot_title[0:4]

    config = config_parser.Config("config.txt")

    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    plot_dir = Path(__file__).resolve().parents[2]/plot_folder
    seq_dir = plot_dir/seq_ID

    if "Layer" in plot_title:
        if "Average" in plot_title:
            plot_path = plot_dir/"avg_att_align_layers.png"
        else:
            plot_path = seq_dir/f"{seq_ID}_att_align_layers.png"

    if plot_path.is_file():
        log.logger.warning(
            f"A file with the same path already exists: {plot_path}\n"
            "The plot will not be saved."
        )
        return None

    fig, ax = plt.subplots()
    ax.set_title(plot_title)

    if type(attention) is np.ndarray:
        if len(attention.shape) == 1:
            ax.bar(list(range(1, len(attention)+1)), attention)

    plt.savefig(plot_path)
    plt.close()


def plot_distance_and_contact(
    distance_map: np.ndarray,
    norm_contact_map: np.ndarray,
    seq_dir: Path,
) -> None:
    """
    Plot the distance map and the normalized contact map side by side.

    Parameters
    ----------
    distance_map : np.ndarray
        The distance in Angstroms between each couple of amino acids in the
        peptide chain.
    norm_contact_map : np.ndarray
        distance_map but in a scale between 0 and 1.
    seq_dir : Path
        The path to the folder containing the plots relative to the peptide
        chain.

    Returns
    -------
    None

    """
    seq_ID = seq_dir.stem
    plot_path = seq_dir/f"{seq_ID}_distance_and_contact.png"

    if plot_path.is_file():
        return None

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(121)
    ax1.set_title(f"{seq_ID}\nDistance Map")
    im1 = ax1.imshow(distance_map, cmap='Blues')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    ax2 = fig.add_subplot(122)
    ax2.set_title(f"{seq_ID}\nNormalized Contact Map")
    im2 = ax2.imshow(norm_contact_map, cmap='Blues')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    plt.savefig(plot_path)
    plt.close()


def plot_heatmap(
    attention: pd.DataFrame | np.ndarray,
    plot_title: str,
) -> None:
    """
    Plot sns.heatmap.

    Parameters
    ----------
    attention : pd.DataFrame | np.ndarray
        Any data structure with shape (number_of_layers, number_of_heads).
    plot_title : str

    Returns
    -------
    None

    """
    seq_ID = plot_title[0:4]

    config = config_parser.Config("config.txt")

    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    plot_dir = Path(__file__).resolve().parents[2]/plot_folder
    seq_dir = plot_dir/seq_ID

    if "Alignment" in plot_title:
        if "Average" in plot_title:
            plot_path = plot_dir/"avg_att_align_heads.png"
        else:
            plot_path = seq_dir/f"{seq_ID}_att_align_heads.png"
    elif "Similarity" in plot_title:
        if "Average" in plot_title:
            plot_path = plot_dir/"avg_att_sim.png"
        else:
            plot_path = seq_dir/f"{seq_ID}_att_sim.png"

    if plot_path.is_file():
        log.logger.warning(
            f"A file with the same path already exists: {plot_path}\n"
            "The plot will not be saved."
        )
        return None

    fig, ax = plt.subplots()
    sns.heatmap(attention)
    ax.set_title(plot_title)

    if type(attention) is np.ndarray:
        if len(attention.shape) == 2:
            xticks = list(range(1, attention.shape[1]+1))
            xticks_labels = list(map(str, xticks))
            yticks = list(range(1, attention.shape[0]+1, 2))
            yticks_labels = list(map(str, yticks))

            ax.set(
                xlabel="Head",
                xticks=xticks,
                xticklabels=xticks_labels,
                ylabel="Layer",
                yticks=yticks,
                yticklabels=yticks_labels,
            )

    plt.savefig(plot_path)
    plt.close()
