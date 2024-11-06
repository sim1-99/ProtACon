"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Process the data relative to the distances between the residues in the peptide
chain. Those data are used to create protein contact maps.

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ProtACon import config_parser
from ProtACon.modules.contact import (
    binarize_contact_map,
    generate_distance_map,
)
from ProtACon.modules.utils import normalize_array

if TYPE_CHECKING:
    from ProtACon.modules.basics import CA_Atom


def main(
    CA_Atoms: tuple[CA_Atom, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate a distance map, a contact map and a binary contact map.

    Parameters
    ----------
    CA_Atoms : tuple[CA_Atom, ...]

    Returns
    -------
    distance_map : np.ndarray
        The distance in Angstroms between each couple of residues in the
        peptide chain.
    norm_contact_map : np.ndarray
        The contact map in a scale between 0 and 1.
    binary_contact_map : np.ndarray
        The contact map binarized using two thresholding criteria.

    """
    config_file_path = Path(__file__).resolve().parents[1]/"config.txt"
    config = config_parser.Config(config_file_path)

    cutoffs = config.get_cutoffs()
    distance_cutoff = cutoffs["DISTANCE_CUTOFF"]
    position_cutoff = cutoffs["POSITION_CUTOFF"]

    distance_map = generate_distance_map(CA_Atoms)
    distance_map_copy = distance_map.copy()

    # set array diagonal to np.nan to avoid divide by 0 error
    distance_map_copy[distance_map_copy == 0.] = np.nan
    contact_map = np.array(1/distance_map_copy)

    norm_contact_map = normalize_array(contact_map)

    binary_contact_map = binarize_contact_map(
        distance_map, distance_cutoff, int(position_cutoff)
    )

    return (
        distance_map,
        norm_contact_map,
        binary_contact_map,
    )
