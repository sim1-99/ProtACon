"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Define the functions for the computation and processing of the contact map of a
peptide chain.

"""
from __future__ import annotations

from typing import TYPE_CHECKING
import math

import numpy as np

if TYPE_CHECKING:
    from ProtACon.modules.basics import CA_Atom


def binarize_contact_map(
    distance_map: np.ndarray,
    distance_cutoff: float,
    position_cutoff: int,
) -> np.ndarray:
    """
    Generate a binary contact map.

    Two criteria, in form of thresholds, are applied to the distance map in
    order to get the binarized contact map:
        - Distance thresholding: 1 is set (i.e., a contact) if two residues
        are at a distance in the native state smaller than distance_cutoff;
        otherwise 0 is set.
        - Position thresholding: we want to keep only the contacts that rise
        from couples of residues which are close in terms of distance in the
        native state, but which are not so close in terms of their position
        along the peptide chain. Therefore, 1 is set (i.e., a contact) if two
        residues are separated by a number of residues larger than
        position_cutoff in the peptide chain; otherwise 0 is set.

    Parameters
    ----------
    distance_map : np.ndarray
        The distance in Angstroms between each couple of residues in the
        peptide chain.
    distance_cutoff : float
        The threshold distance expressed in Angstroms.
    position_cutoff : int
        The threshold position difference between residues in the peptide
        chain.

    Returns
    -------
    binary_contact_map : np.ndarray
        The contact map binarized using two thresholding criteria.

    """
    binary_contact_map = np.where(distance_map <= distance_cutoff, 1.0, 0.0)

    for i in range(binary_contact_map.shape[0]):
        for j in range(binary_contact_map.shape[1]):
            if abs(i-j) < position_cutoff:
                binary_contact_map[i][j] *= 0.
            elif abs(i-j) >= position_cutoff:
                binary_contact_map[i][j] *= 1.

    return binary_contact_map


def distance_between_atoms(
    atom1_coords: np.ndarray,
    atom2_coords: np.ndarray,
) -> float:
    """
    Compute the distance in Angstroms between two atoms.

    Parameters
    ----------
    atom1_coords: np.ndarray
    atom2_coords: np.ndarray

    Returns
    -------
    norm : float
        The distance in Angstroms between two atoms.

    """
    x1 = atom1_coords[0]
    x2 = atom2_coords[0]
    y1 = atom1_coords[1]
    y2 = atom2_coords[1]
    z1 = atom1_coords[2]
    z2 = atom2_coords[2]
    x_distance = x1-x2
    y_distance = y1-y2
    z_distance = z1-z2
    distance = (x_distance**2, y_distance**2, z_distance**2)
    norm = math.sqrt(math.fsum(distance))

    return norm


def generate_distance_map(
    CA_Atoms: tuple[CA_Atom, ...],
) -> np.ndarray:
    """
    Generate a distance map that stores the distance in Angstroms between each
    couple of residues in the peptide chain.

    Parameters
    ----------
    CA_Atoms : tuple[CA_Atom, ...]

    Returns
    -------
    distance_map : np.ndarray
        The distance in Angstroms between each couple of residues in the
        peptide chain.

    """
    distance_map = np.full((len(CA_Atoms), len(CA_Atoms)), np.nan)

    for x, atom_x in enumerate(CA_Atoms):
        for y, atom_y in enumerate(CA_Atoms):
            distance_map[x, y] = distance_between_atoms(
                np.array(atom_x.coords), np.array(atom_y.coords)
            )

    return distance_map
