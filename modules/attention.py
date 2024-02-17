#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention.

This module contains functions for the extraction and processing of attention
from the ProtBert model.
"""


__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from utils import dict_3_to_1


def get_sequence_to_tokenize(CA_Atoms: tuple) -> str:
    """
    Return a string of amino acids in a format suitable for tokenization.

    The function takes the name attribute of the CA_Atom objects in the tuple,
    translate them from multiple letter to single letter amino acid codes and
    append them to a single string, ready to be tokenized.

    Parameters
    ----------
    CA_Atoms : tuple

    Returns
    -------
    sequence : str
        sequence of amino acids

    """
    sequence = ""
    for atom in CA_Atoms:
        sequence = sequence + str(dict_3_to_1[atom.name]) + " "

    return sequence
