"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Define the plot functions of ProtACon (attention matrices, attention heatmaps,
contact maps, etc.).

"""
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ProtACon import config_parser
from ProtACon.modules.miscellaneous import dict_1_to_3
from ProtACon.modules.utils import Logger


log = Logger("mylog").get_logger()


def find_best_nrows(
    n_am_ac: int,
) -> int:
    """
    Find the adequate number of rows to use in plt.subplots.

    Parameters
    ----------
    n_am_ac : int

    Raises
    ------
    ValueError
        If the amino acids in the chain are more than twenty.

    Returns
    -------
    nrows : int
        The number of rows to use in plt.subplots.

    """
    ncols = 4
    quotient = n_am_ac/ncols

    if quotient > 5:
        raise ValueError("Found more than twenty amino acids")

    if quotient > int(quotient):
        nrows = int(quotient)+1
    elif quotient == int(quotient):
        nrows = int(quotient)

    return nrows


def plot_attention_matrices(
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

    config_file_path = Path(__file__).resolve().parents[2]/"config.txt"
    config = config_parser.Config(config_file_path)

    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    seq_dir = Path(__file__).resolve().parents[2]/plot_folder/seq_ID

    if type(attention) is torch.Tensor:
        nrows = 1
        ncols = 1
        plot_path = seq_dir/f"{seq_ID}_att_model_avg.png"
    elif len(attention) == 30:
        if len(attention[0].size()) == 2:
            nrows = 6
            ncols = 5
            plot_path = seq_dir/f"{seq_ID}_att_layer_avg.png"
        elif len(attention[0].size()) == 3:
            nrows = 4
            ncols = 4
            layer_number = int(plot_title[-2:])
            plot_path = seq_dir/f"{seq_ID}_att_layer_{layer_number}.png"

    if plot_path.is_file():
        log.logger.warning(
            f"A file with the same path already exists: {plot_path}\n"
            "The plot will not be saved."
        )
        return None

    head_idx = 0

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    fig.suptitle(plot_title, fontsize=18)
    for row in range(nrows):
        for col in range(ncols):
            if type(attention) is torch.Tensor:
                img = attention.numpy()
                plt.imshow(img, cmap='Blues')
            elif len(attention) == 30:
                if len(attention[0].size()) == 2:
                    img = attention[head_idx].numpy()
                    axes[row, col].set_title(f"Layer {head_idx+1}")
                elif len(attention[0].size()) == 3:
                    img = attention[layer_number-1][head_idx].numpy()
                    axes[row, col].set_title(f"Head {head_idx+1}")
                axes[row, col].set_xticks([])
                axes[row, col].imshow(img, cmap='Blues')
            head_idx += 1

    plt.savefig(plot_path)
    plt.close()


def plot_attention_to_amino_acids_alone(
    attention_to_amino_acids: torch.Tensor,
    amino_acids: list[str],
    plot_dir: Path,
    plot_title: str,
) -> None:
    """
    Plot the attention heatmaps for more amino acids in separate plots.

    The heatmaps are filled with the values of attention given to each amino
    acid by each attention head.
    This function is used to plot the heatmaps representing the percentage of
    each head's attention given to each amino acid. Each heatmap is plotted
    separately and saved in a dedicated file, because the percentage
    represented makes sense only within one heatmap.

    Parameters
    ----------
    attention_to_amino_acids : torch.Tensor
        Tensor with shape (n_am_ac, n_layers, n_heads), storing the attention
        given to each amino acid by each attention head.
    amino_acids : list[str]
        The single letter codes of the amino acids in the peptide chain or in
        the set of peptide chains.
    plot_dir : Path
        The path to the folder where to store the plots.
    plot_title : str

    Raises
    ------
    ValueError
        If plt.subplots has too many rows with respect to the number of amino
        acids in the chain.

    Returns
    -------
    None

    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = [plot_dir/f"PH_att_to_{aa}.png" for aa in amino_acids]

    for data, amino_acid, path in zip(
        attention_to_amino_acids, amino_acids, plot_paths
    ):
        if path.is_file():
            log.logger.warning(
                f"A file with the same path already exists: {path}\n"
                "The plot will not be saved."
            )
            continue
        fig, ax = plt.subplots()
        sns.heatmap(data.numpy())

        xticks = list(range(1, attention_to_amino_acids.size(dim=2)+1))
        xticks_labels = list(map(str, xticks))
        yticks = list(range(1, attention_to_amino_acids.size(dim=1)+1, 2))
        yticks_labels = list(map(str, yticks))

        ax.set(
            title=f"{plot_title}\n{dict_1_to_3[amino_acid][1]} "
            f"({amino_acid})",
            xlabel="Head",
            xticks=xticks,
            xticklabels=xticks_labels,
            ylabel="Layer",
            yticks=yticks,
            yticklabels=yticks_labels,
        )
        ax.collections[0].colorbar.set_label("%", rotation="horizontal")

        plt.savefig(path)
        plt.close()


def plot_attention_to_amino_acids_together(
    attention_to_amino_acids: torch.Tensor,
    amino_acids: list[str],
    plot_path: Path,
    plot_title: str,
) -> None:
    """
    Plot the attention heatmaps for more amino acids together.

    The heatmaps are filled with the values of attention given to each amino
    acid by each attention head.
    This function is used to plot the heatmaps representing the percentage of
    total attention. They must be shown all together in one file, because the
    percentage represented makes sense only when all the heatmaps relative to
    the different amino acids are shown.
    This function is also used to plot the heatmaps representing the absolute
    attention given to each amino acid in the single peptide chains.

    Parameters
    ----------
    attention_to_amino_acids : torch.Tensor
        Tensor with shape (n_am_ac, n_layers, n_heads), storing the attention
        given to each amino acid by each attention head.
    amino_acids : list[str]
        The single letter codes of the amino acids in the peptide chain or in
        the set of peptide chains.
    plot_path : Path
        The path where to store the plot.
    plot_title : str

    Raises
    ------
    ValueError
        If plt.subplots has too many rows with respect to the number of amino
        acids in the chain.

    Returns
    -------
    None

    """
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
        nrows=nrows, ncols=ncols, figsize=(20, 20), constrained_layout=True
    )
    fig.suptitle(plot_title, fontsize=18)

    if nrows == 1:
        # otherwise axes is a 1D array and cannot iterate on rows and cols
        axes = np.reshape(axes, (nrows, ncols))
        # make the plot more readable
        fig.set_size_inches(20, 10)

    for row in range(nrows):
        for col in range(ncols):
            img = attention_to_amino_acids[amino_acid_idx].numpy()
            sns.heatmap(img, ax=axes[row, col])

            axes[row, col].set(
                title=f"{dict_1_to_3[amino_acids[amino_acid_idx]][1]} "
                f"({amino_acids[amino_acid_idx]})",
                xlabel="Head",
                xticks=xticks,
                xticklabels=xticks_labels,
                ylabel="Layer",
                yticks=yticks,
                yticklabels=yticks_labels,
            )
            if "Percentage" in plot_title:
                axes[row, col].collections[0].colorbar.set_label(
                    "%", rotation="horizontal"
                )

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
    plot_path: Path,
    plot_title: str,
) -> None:
    """
    Plot a barplot.

    Parameters
    ----------
    attention : np.ndarray
        Any data structure with shape (n_layers).
    plot_path : Path
        The path where to store the plot.
    plot_title : str

    Returns
    -------
    None

    """
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
    plot_path: Path,
) -> None:
    """
    Plot the distance map and the normalized contact map side by side.

    Parameters
    ----------
    distance_map : np.ndarray
        The distance in Angstroms between each couple of residues in the
        peptide chain.
    norm_contact_map : np.ndarray
        distance_map but in a scale between 0 and 1.
    plot_path : Path
        The path where to store the plot.

    Returns
    -------
    None

    """
    seq_ID = plot_path.parent.stem

    if plot_path.is_file():
        log.logger.warning(
            f"A file with the same path already exists: {plot_path}\n"
            "The plot will not be saved."
        )
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
    data: pd.DataFrame | np.ndarray,
    plot_path: Path,
    plot_title: str,
) -> None:
    """
    Plot sns.heatmap.

    Parameters
    ----------
    attention : pd.DataFrame | np.ndarray
    plot_path : Path
        The path where to store the plot.
    plot_title : str

    Returns
    -------
    None

    """
    if plot_path.is_file():
        log.logger.warning(
            f"A file with the same path already exists: {plot_path}\n"
            "The plot will not be saved."
        )
        return None

    fig, ax = plt.subplots()
    sns.heatmap(data)
    ax.set_title(plot_title)

    if type(data) is np.ndarray:
        if len(data.shape) == 2:
            xticks = list(range(1, data.shape[1]+1))
            xticks_labels = list(map(str, xticks))
            yticks = list(range(1, data.shape[0]+1, 2))
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
