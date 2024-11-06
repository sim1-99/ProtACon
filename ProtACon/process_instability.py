#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process instability.

This script processes the data on the instability index bewteen the amino acids in the
peptide chain. Indices are used to create protein instability maps,
w.r.t. Bio.SeqUtils.ProtParamData.DIWV.
"""

from __future__ import annotations

__author__ = 'Renato Eliasy'
__email__ = 'renatoeliasy@gmail.com'

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ProtACon import config_parser
from ProtACon.modules.contact import (
    binarize_instability_map,
    generate_instability_map,
)
from ProtACon import process_contact

if TYPE_CHECKING:
    from ProtACon.modules.basics import CA_Atom


def main(
    CA_Atoms: tuple[CA_Atom, ...]
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate an instability map, an instability-contact map and a binary
    instability map.

    Parameters
    ----------
    CA_Atoms : tuple[CA_Atom, ...]

    Returns
    -------
    instability_map : np.ndarray
        stores the instability index - collected by the DIWV dictionary -
        between each couple of residues in the peptide chain
    contact_instability_map : np.ndarray
        stores how much each residue is in a stable link to all the others that
        are linked in the respective contact map
    binary_instability_map : np.ndarray
        contact map binarized using two possible thresholding criteria

    """
    config_file_path = Path(__file__).resolve().parents[1]/"config.txt"
    config = config_parser.Config(config_file_path)

    cutoffs = config.get_cutoffs()
    instability_cutoff = cutoffs["INSTABILITY_CUTOFF"]
    stability_cutoff = cutoffs["STABILITY_CUTOFF"]

    instability_map = generate_instability_map(CA_Atoms)

    binarized_instability_map = binarize_instability_map(
        inst_map=instability_map,
        stability_cutoff=stability_cutoff,
    )

    *_, contact_binarized_map = process_contact.main(CA_Atoms)

    binarized_contact_instability_map = binarize_instability_map(
        inst_map=instability_map,
        base_map=contact_binarized_map,
        stability_cutoff=stability_cutoff,
    )

    return (
        instability_map,
        binarized_instability_map,
        binarized_contact_instability_map,
    )
