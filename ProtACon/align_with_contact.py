"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

This script combines other scripts for the computation of the attention
alignment of the contact map of one protein. Other meaningful quantities, such
as pairwise attention similarity, are computed too. The user can also choose if
to plot those quantities for every single protein in the set.

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ProtACon import config_parser
from ProtACon import process_attention
from ProtACon import process_contact
from ProtACon import plotting
from ProtACon.modules.miscellaneous import all_amino_acids

if TYPE_CHECKING:
    from ProtACon.modules.miscellaneous import CA_Atom


def main(
    attention: tuple[torch.Tensor, ...],
    CA_Atoms: tuple[CA_Atom, ...],
    chain_amino_acids: list[str],
    att_to_am_ac: torch.Tensor,
    seq_ID: str,
    save_opt: str,
) -> tuple[
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
    att_to_am_ac : torch.Tensor
        Tensor with shape (len(all_amino_acids), number_of_layers,
        number_of_heads), storing the absolute attention given to each type of
        amino acid by each attention head.
    seq_ID : str
        The alphanumerical code representing uniquely the peptide chain.
    save_opt : str
        One between ('none', 'plot', 'csv', 'both'). If 'plot' or 'both', save
        the plots of every single chain.

    Returns
    -------
    head_att_align : np.ndarray
        Array with shape (number_of_layers, number_of_heads), storing how much
        attention aligns with indicator_function for each attention matrices.
    layer_att_align : np.ndarray
        Array with shape (number_of_layers), storing how much attention aligns
        with indicator_function for each average attention matrix computed
        independently over each layer.

    """
    config = config_parser.Config("config.txt")

    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    plot_dir = Path(__file__).resolve().parents[1]/plot_folder

    save_if = ("plot", "both")

    # remove zero tensors from att_to_am_ac
    nonzero_indices = [
        all_amino_acids.index(type) for type in chain_amino_acids
    ]
    att_to_am_ac = torch.index_select(
        att_to_am_ac, 0, torch.tensor(nonzero_indices)
    )

    distance_map, norm_contact_map, binary_contact_map = process_contact.main(
        CA_Atoms
    )

    att_sim_df, att_avgs, head_att_align, layer_att_align = \
        process_attention.main(
            attention, att_to_am_ac, binary_contact_map, chain_amino_acids
        )

    if save_opt in save_if:
        seq_dir = plot_dir/seq_ID
        seq_dir.mkdir(parents=True, exist_ok=True)
        plotting.plot_on_chain(
            distance_map, norm_contact_map, binary_contact_map, attention,
            att_avgs, att_to_am_ac, att_sim_df, head_att_align,
            layer_att_align, seq_dir, chain_amino_acids
        )

    return (
        head_att_align,
        layer_att_align,
    )
