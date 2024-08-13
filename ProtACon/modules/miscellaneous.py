"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

This module defines:
    - the dictionaries for translating from multiple letter to single letter
      amino acid codes, and vice versa
    - the implementation of the CA_Atom class
    - functions for extracting information from ProtBert and from PDB objects

"""
import logging

from Bio.PDB.Structure import Structure
from transformers import BertModel, BertTokenizer
import torch


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

    def __init__(
        self,
        name: str,
        idx: int,
        coords: list[float],
    ):
        """
        Contructor of the class.

        Parameters
        ----------
        name : str
            The name of the amino acid.
        idx : int
            The position of the amino acid along the chain.
        coords : list[float]
            The x-, y- and z- coordinates of the CA atom of the amino acid.

        """
        self.name = name
        self.idx = idx
        self.coords = coords


def extract_CA_Atoms(
    structure: Structure,
) -> tuple[CA_Atom, ...]:
    """
    Extract the CA atoms from the peptide chain and put them in a tuple as
    CA_Atom objects.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The object containing information about each atom of the peptide chain.

    Returns
    -------
    CA_Atoms_tuple : tuple[CA_Atom, ...]

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
                logging.warning(" Found and discarded ligand in position: "
                                f"{residue_idx}")
    CA_Atoms_tuple = tuple(CA_Atoms_list)

    return CA_Atoms_tuple


def get_model_structure(
    raw_attention: tuple[torch.Tensor, ...],
) -> tuple[
    int,
    int,
]:
    """
    Return the number of heads and the number of layers of ProtBert.

    Parameters
    ----------
    raw_attention : tuple[torch.Tensor, ...]
        The attention from the model, including the attention relative to
        tokens [CLS] and [SEP].

    Returns
    -------
    number_of_heads : int
        The number of heads of ProtBert.
    number_of_layers : int
        The +number of layers of ProtBert.

    """
    layer_structure = raw_attention[0].shape
    get_model_structure.number_of_heads = layer_structure[1]
    get_model_structure.number_of_layers = len(raw_attention)

    return (
        get_model_structure.number_of_heads,
        get_model_structure.number_of_layers
    )


def get_sequence_to_tokenize(
    CA_Atoms: tuple[CA_Atom, ...],
) -> str:
    """
    Return a string of amino acids in a format suitable for tokenization. The
    function takes the name attribute of the CA_Atom objects in the tuple,
    translate them from multiple letter to single letter amino acid codes and
    append them to a single string, ready to be tokenized.

    Parameters
    ----------
    CA_Atoms : tuple[CA_Atom, ...]

    Returns
    -------
    sequence : str
        The sequence of amino acids.

    """
    sequence = ""
    for atom in CA_Atoms:
        sequence = sequence + atom.name + " "

    return sequence


def get_types_of_amino_acids(
    tokens: list[str],
) -> list[str]:
    """
    Return a list with the types of the residues present in the peptide chain.

    Parameters
    ----------
    tokens : list[str]
        The tokens used by the model, cleared of the tokens [CLS] and [SEP].

    Returns
    -------
    types_of_amino_acids : list[str]
        The single letter amino acid codes of the amino acid types in the
        peptide chain.

    """
    types_of_amino_acids = list(dict.fromkeys(tokens))

    return types_of_amino_acids


def load_model(
    model_name: str,
) -> tuple[
    BertModel,
    BertTokenizer,
]:
    """
    Load the model and the tokenizer specified by model_name.

    Parameters
    ----------
    model_name : str

    Returns
    -------
    model : BertModel
    tokenizer : BertTokenizer

    """
    load_model.model = BertModel.from_pretrained(
        model_name, output_attentions=True)
    load_model.tokenizer = BertTokenizer.from_pretrained(
        model_name, do_lower_case=False)

    return (
        load_model.model,
        load_model.tokenizer
    )
