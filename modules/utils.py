#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils.

This module contains:
    - the implementation of the CA_Atom class
    - the dictionaries for translating from multiple letter to single letter
      amino acid codes, and vice versa
    - the function for reading the .pdb files
    - the implementation of a timer
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

import config_parser

from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from functools import reduce
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union

from contextlib import contextmanager
from datetime import datetime
import logging


@contextmanager
def Timer(description: str):
    """
    Timer.

    Parameters
    ----------
    description : str
        text to print before variable message

    Returns
    -------
    None.

    """
    start = datetime.now()
    try:
        yield
    finally:
        end = datetime.now()
        timedelta = end-start
        message = (f"{description}, started: {start}, ended: {end}, elapsed:"
                   f"{timedelta}")
        logging.warning(message)


dict_1_to_3 = {
    "A": ["ALA", "Alanine"],
    "R": ["ARG", "Arginine"],
    "N": ["ASN", "Asparagine"],
    "D": ["ASP", "Aspartic Acid"],
    "C": ["CYS", "Cysteine"],
    "Q": ["GLN", "Glutamine"],
    "E": ["GLU", "Glutamic Acid"],
    "G": ["GLY", "Glycine"],
    "H": ["HIS", "Histidine"],
    "I": ["ILE", "Isoleucine"],
    "L": ["LEU", "Leucine"],
    "K": ["LYS", "Lysine"],
    "M": ["MET", "Methionine"],
    "F": ["PHE", "Phenylalanine"],
    "P": ["PRO", "Proline"],
    "S": ["SER", "Serine"],
    "T": ["THR", "Threonine"],
    "W": ["TRP", "Tryptophan"],
    "Y": ["TYR", "Tyrosine"],
    "V": ["VAL", "Valine"],
}

dict_3_to_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V"
}


class CA_Atom:
    """A class to represent CA atoms of amino acids."""

    def __init__(self, name: str, idx: int, coords: list):
        """
        Contructor of the class.

        Parameters
        ----------
        name : str
            name of the amino acid
        idx : int
            position of the amino acid along the chain
        coords : list
            x-, y- and z- coordinates of the CA atom of the amino acid

        """
        self.name = name
        self.idx = idx
        self.coords = coords


def average_maps_together(list_of_maps: List[Union[pd.DataFrame, np.ndarray]]
                          ) -> Union[pd.DataFrame, np.ndarray]:
    """
    Average together the maps (tensors or arrays) contained in a list.

    Parameters
    ----------
    list_of_maps : List[Union[pd.DataFrame, np.ndarray]]
        contains the maps to be averaged together

    Returns
    -------
    average_map : pd.DataFrame or np.ndarray

    """
    if type(list_of_maps[0]) is pd.DataFrame:
        average_map = reduce(lambda x, y: x.add(y, fill_value=0), list_of_maps)
        average_map.div(len(list_of_maps))
        """  # TODO: remove if not use
        average_map = torch.sum(
            torch.stack(list_of_maps), dim=0)/len(list_of_maps)
        """
        return average_map

    if type(list_of_maps[0]) is np.ndarray:
        average_map = np.sum(np.stack(list_of_maps), axis=0)/len(list_of_maps)
        return average_map


def extract_CA_Atoms(structure) -> tuple:
    """
    Get all CA atoms.

    Extract CA atoms from the peptide chain and put them in a tuple as CA_Atom
    objects.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        object containing information about each atom of the peptide chain

    Returns
    -------
    CA_Atoms_tuple : tuple

    """
    chain = structure[0]["A"]
    residues = chain.get_list()
    CA_Atoms_list = []

    for residue_idx, residue in enumerate(residues):
        for atom in residue:
            if (atom.get_name() == "CA" and
                    residue.get_resname() in dict_3_to_1):
                CA_Atoms_list.append(CA_Atom(
                    name=dict_3_to_1[residue.get_resname()],
                    idx=residue_idx,
                    coords=atom.get_coord()))
                break
            elif atom.get_name() == "CA":
                logging.warning("Found and discarded ligand in position: "
                                f"{residue_idx}")
    CA_Atoms_tuple = tuple(CA_Atoms_list)

    return CA_Atoms_tuple


def get_model_structure(raw_attention: tuple):
    """
    Return the number of heads and the number of layers of ProtBert.

    Parameters
    ----------
    raw_attention : tuple
        contains tensors that store the attention from the model, including the
        attention relative to tokens [CLS] and [SEP]

    Returns
    -------
    number_of_heads : int
        number of heads of ProtBert
    number_of_layers : int
        number of layers of ProtBert

    """
    layer_structure = raw_attention[0].shape
    get_model_structure.number_of_heads = layer_structure[1]
    get_model_structure.number_of_layers = len(raw_attention)

    return (get_model_structure.number_of_heads,
            get_model_structure.number_of_layers)


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
        sequence = sequence + atom.name + " "

    return sequence


def get_types_of_amino_acids(tokens: list) -> list:
    """
    Return a list with the types of the residues present in the peptide chain.

    Parameters
    ----------
    tokens : list
        contains strings which are the tokens used by the model, cleared of the
        tokens [CLS] and [SEP]

    Returns
    -------
    types_of_amino_acids : list
        contains strings with single letter amino acid codes of the amino acid
        types in the peptide chain

    """
    types_of_amino_acids = list(dict.fromkeys(tokens))

    return types_of_amino_acids


def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy array.

    Parameters
    ----------
    array : np.ndarray

    Returns
    -------
    norm_array : np.ndarray

    """
    if True in np.isnan(array):
        array_max, array_min = np.nanmax(array), np.nanmin(array)
    else:
        array_max, array_min = np.max(array), np.min(array)
    norm_array = (array - array_min)/(array_max - array_min)

    return norm_array


def read_pdb_file(seq_ID: str):
    """
    Download the .pdb file of the sequence ID to get its structure.

    Parameters
    ----------
    seq_ID : str
        alphanumerical code representing uniquely one peptide chain

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        object containing information about each atom of the peptide chain

    """
    config = config_parser.Config("config.txt")
    paths = config.get_paths()
    pdb_folder = paths["PDB_FOLDER"]
    pdb_dir = Path(__file__).parent.parent/pdb_folder

    pdb_import = PDBList()
    pdb_file = pdb_import.retrieve_pdb_file(
        pdb_code=seq_ID, file_format="pdb", pdir=pdb_dir)

    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(seq_ID, pdb_file)

    return structure
