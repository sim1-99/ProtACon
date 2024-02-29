#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process contact.

This script processes the data on the distances bewteen the amino acids in the
peptide chain. Distances are used to create protein contact maps.
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from modules.contact import binarize_contact_map, generate_distance_map
from modules.utils import normalize_array
import numpy as np


def main(CA_Atoms: tuple) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Generate a distance map, a contact map and a binary contact map.

    Parameters
    ----------
    CA_Atoms : tuple

    Returns
    -------
    distance_map : np.ndarray
        it shows the distance - expressed in Angstroms - between each couple of
        amino acids in the peptide chain
    norm_contact_map : np.ndarray
        it shows how much each amino acid is close to all the others, in a
        scale between 0 and 1
    binary_contact_map : np.ndarray
        contact map binarized using two thresholding criteria

    """
    distance_map = generate_distance_map(CA_Atoms)
    distance_map_copy = distance_map.copy()

    # set array diagonal to 0 to avoid divide by 0 error
    distance_map_copy[distance_map_copy == 0.] = np.nan
    contact_map = np.array(1/distance_map_copy)
    del distance_map_copy

    norm_contact_map = normalize_array(contact_map)

    distance_cutoff = 8.0
    position_cutoff = 6
    binary_contact_map = binarize_contact_map(
        distance_map, distance_cutoff, position_cutoff)

    return distance_map, norm_contact_map, binary_contact_map
