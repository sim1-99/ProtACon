#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contact.

This module contains functions for the computation and processing of the
contact map of a peptide chain.
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

import math
from modules.utils import CA_Atom


def distance_between_CA(atom_1: CA_Atom, atom_2: CA_Atom) -> float:
    """
    Compute the distance - expressed in Angstroms - between the two atoms.

    Parameters
    ----------
    atom_1 : CA_Atom
    atom_2 : CA_Atom

    Returns
    -------
    norm : float
        distance in Angstroms between the two atoms

    """
    x1 = atom_1[0]
    x2 = atom_2[0]
    y1 = atom_1[1]
    y2 = atom_2[1]
    z1 = atom_1[2]
    z2 = atom_2[2]
    x_distance = x1-x2
    y_distance = y1-y2
    z_distance = z1-z2
    distance = (x_distance**2, y_distance**2, z_distance**2)
    norm = math.sqrt(math.fsum(distance))

    return norm
