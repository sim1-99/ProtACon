"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Compute the attention alignment with the contact map of one peptide chain. The
pairwise attention similarity is computed, too. The user can also choose if to
plot and save those quantities for every single protein in the set.

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ProtACon import config_parser
from ProtACon import process_contact
from ProtACon import plotting
from ProtACon.modules.attention import (
    average_matrices_together,
    compute_att_align_on_heads,
    compute_att_align_on_layers,
    compute_attention_similarity,
)
from ProtACon.modules.basics import all_amino_acids

if TYPE_CHECKING:
    from ProtACon.modules.basics import CA_Atom


def main(
    attention: tuple[torch.Tensor, ...],
    CA_atoms: tuple[CA_Atom, ...],
    chain_amino_acids: list[str],
    att_to_aa: torch.Tensor,
    seq_ID: str,
    save_opt: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Compute the attention alignment with the contact map for one chain.

    Compute the alignment of the attention matrices with the contact map for
    the peptide chain identified with seq_ID. Pairwise attention similarity is
    computed, too. Both those quantities can be plotted and saved.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention matrices returned by the model, cleared of the attention
        related to the tokens [CLS] and [SEP].
    CA_atoms: tuple[CA_Atom, ...]
        The alpha carbon atoms of the residues in the peptide chain.
    chain_amino_acids : list[str]
        The single letter codes of the amino acid in the peptide chain.
    att_to_aa : torch.Tensor
        Tensor with shape (len(all_amino_acids), n_layers, n_heads), storing
        the absolute attention given to each amino acid by each attention head.
    seq_ID : str
        The alphanumerical code representing uniquely the protein.
    save_opt : str
        One between ('none', 'plot', 'csv', 'both'). If 'plot' or 'both', save
        the plots of every single chain.

    Returns
    -------
    head_att_align : np.ndarray
        Array with shape (n_layers, n_heads), storing how much attention aligns
        with indicator_function for each attention matrices.
    layer_att_align : np.ndarray
        Array with shape (n_layers), storing how much attention aligns with
        indicator_function for each average attention matrix computed
        independently over each layer.

    """
    config_file_path = Path(__file__).resolve().parents[1]/"config.txt"
    config = config_parser.Config(config_file_path)

    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    plot_dir = Path(__file__).resolve().parents[1]/plot_folder

    save_if = ("plot", "both")

    # remove zero tensors from att_to_aa
    nonzero_indices = [
        all_amino_acids.index(type) for type in chain_amino_acids
    ]
    att_to_aa = torch.index_select(
        att_to_aa, 0, torch.tensor(nonzero_indices)
    )

    att_avgs = average_matrices_together(attention)

    distance_map, norm_contact_map, binary_contact_map = process_contact.main(
        CA_atoms
    )

    head_att_align = compute_att_align_on_heads(
        attention=attention,
        indicator_function=binary_contact_map,
    )
    layer_att_align = compute_att_align_on_layers(
        attention=att_avgs[:-1],
        indicator_function=binary_contact_map,
    )

    if save_opt in save_if:
        seq_dir = plot_dir/seq_ID
        seq_dir.mkdir(parents=True, exist_ok=True)

        att_sim_df = compute_attention_similarity(att_to_aa, chain_amino_acids)
        plotting.plot_on_chain(
            distance_map, norm_contact_map, binary_contact_map, attention,
            att_avgs, att_to_aa, att_sim_df, head_att_align, layer_att_align,
            seq_dir, chain_amino_acids
        )

    return (
        head_att_align,
        layer_att_align,
    )
