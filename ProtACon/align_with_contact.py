"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

This script combines other scripts for the computation of the attention
alignment of the contact map of one protein. Other meaningful quantities, such
as pairwise attention similarity, are computed too. In case of a set of
proteins, those quantities can be averaged over it. The user can also choose if
to plot and save all the plots of every single protein in the set.

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from ProtACon import config_parser
from ProtACon import process_attention
from ProtACon import process_contact
from ProtACon import plotting

if TYPE_CHECKING:
    from ProtACon.modules.miscellaneous import CA_Atom


def main(
    attention: tuple[torch.Tensor, ...],
    CA_Atoms: tuple[CA_Atom, ...],
    chain_amino_acids: list[str],
    attention_to_amino_acids: torch.Tensor,
    seq_ID: str,
    save_single=False,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """
    Run the main function of align_with_contact.py. It computes the attention
    alignment with the contact map and other quantities for the peptide chain
    identified with seq_ID.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention from the model, cleared of the attention relative to
        tokens [CLS] and [SEP].
    CA_Atoms: tuple[CA_Atom, ...]
    chain_amino_acids : list[str]
        The single letter codes of the amino acid types in the peptide chain.
    attention_to_amino_acids : torch.Tensor
        Tensor with shape (number_of_amino_acids, number_of_layers,
        number_of_heads), storing the absolute attention given to each amino
        acid by each attention head.
    seq_ID : str
        The alphanumerical code representing uniquely the peptide chain.
    save_single : bool, default is False
        If True, run plotting.main() and save the plots.

    Returns
    -------
    att_sim_df : pd.DataFrame
        The attention similarity between each couple of amino acids.
    head_att_align : np.ndarray
        Array with shape (number_of_layers, number_of_heads), storing how much
        attention aligns with indicator_function for each attention masks.
    layer_att_align : np.ndarray
        Array with shape (number_of_layers), storing how much attention aligns
        with indicator_function for each average attention mask computed
        independently over each layer.

    """
    config = config_parser.Config("config.txt")

    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    plot_dir = Path(__file__).resolve().parents[1]/plot_folder

    save_if = ("plot", "both")

    distance_map, norm_contact_map, binary_contact_map = process_contact.main(
        CA_Atoms
    )

    att_sim_df, attention_avgs, attention_align = process_attention.main(
        attention, attention_to_amino_acids, binary_contact_map,
        chain_amino_acids
    )

    if save_single is True:
        plotting.plot_on_chain(
            distance_map, norm_contact_map, binary_contact_map, attention,
            attention_avgs, attention_to_amino_acids, att_sim_df,
            attention_align, seq_dir, chain_amino_acids
        )

    return (
        att_sim_df,
        attention_align[0],
        attention_align[1],
    )
