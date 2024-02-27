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
    contact_map : np.ndarray
        it shows how much each amino acid is close to all the others, in a
        scale between 0 and 1
    binary_contact_map : np.ndarray
        contact map binarized using two thresholding criteria

    """
    distance_map = generate_distance_map(CA_Atoms)
    contact_map = np.array(1/distance_map)

    distance_cutoff = 8.0
    position_cutoff = 6
    binary_contact_map = binarize_contact_map(
        distance_map, distance_cutoff, position_cutoff)

    return distance_map, contact_map, binary_contact_map
