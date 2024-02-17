#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils.

This module contains:
    - the implementation of the CA_Atom class
    - the dictionaries for translating from multiple letter to single letter
      amino acid codes, and vice versa
    - the function for reading the .pdb files
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

import Bio.PDB.Structure.Structure
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
# from pathlib import Path
# from Bio.PDB.Polypeptide import PPBuilder


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


def extract_CA_Atoms(structure: Bio.PDB.Structure.Structure) -> tuple:
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
        CA_Atom objects

    """
    chain = structure[0]["A"]
    residues = chain.get_list()
    CA_Atoms_list = []

    for residue_idx, residue in enumerate(residues):
        for atom in residue:
            if atom.get_name() == "CA":
                CA_Atoms_list.append(CA_Atom(
                    name=dict_3_to_1[residue.get_resname()],
                    idx=residue_idx,  # residue.get_id()[1],
                    coords=atom.get_coord()))
    CA_Atoms_tuple = tuple(CA_Atoms_list)
    del CA_Atoms_list

    return CA_Atoms_tuple


def read_pdb_file(seq_ID: str) -> Bio.PDB.Structure.Structure:
    """
    Download the .pdb file of the sequence ID to get its structure.

    Parameters
    ----------
    seq_ID : str
        Alphanumerical code representing uniquely one peptide chain.

    Returns
    -------
    main sequence : str
        amino acid symbols (one letter)

    structure : Bio.PDB.Structure.Structure
        object containing information about each atom of the peptide chain

    """
    pdb_import = PDBList()
    pdb_file = pdb_import.retrieve_pdb_file(pdb_code=seq_ID, file_format="pdb")

    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(seq_ID, pdb_file)

    # peptides = PPBuilder().build_peptides(structure)
    # sequences = peptides.get_sequence()
    # main_sequence = sequences[0]

    return structure
