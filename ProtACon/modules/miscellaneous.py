"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

This module defines:
    - the dictionaries for translating from multiple letter to single letter
      amino acid codes, and vice versa
    - a list with all the possible types of amino acids
    - the implementation of the CA_Atom class
    - functions for extracting information from ProtBert and from PDB objects

"""
import random

from Bio.PDB.Structure import Structure
from rcsbsearchapi import rcsb_attributes as attrs  # type: ignore
from rcsbsearchapi.search import AttributeQuery  # type: ignore
from transformers import BertModel, BertTokenizer  # type: ignore
import torch

from ProtACon.modules.utils import Logger


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
    "VAL": "V",
}

all_amino_acids = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R",
    "S", "T", "V", "W", "Y"
]

log = Logger("cheesecake").get_logger()


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
    chains = structure[0].get_list()
    chain = chains[0]
    log.logger.info(f"Taking chain: [purple]{chain.get_id()}")

    residues = chain.get_list()
    CA_Atoms_list = []

    for residue_idx, residue in enumerate(residues):
        for atom in residue:
            if (
                atom.get_name() == "CA" and
                residue.get_resname() in dict_3_to_1
            ):
                CA_Atoms_list.append(
                    CA_Atom(
                        name=dict_3_to_1[residue.get_resname()],
                        idx=residue_idx,
                        coords=atom.get_coord()
                    )
                )
                break
            elif atom.get_name() == "CA":
                log.logger.info(
                    f"Found and discarded ligand in position: {residue_idx}"
                )
    CA_Atoms_tuple = tuple(CA_Atoms_list)

    return CA_Atoms_tuple


def fetch_pdb_entries(
    max_length: int,
    n_results: int,
    stricter_search: bool = False,
) -> list[str]:
    """
    Fetch PDB entries based on returning proteins, and on the maximum number of
    peptides in the structure. Keep only the number of results specified by
    n_results.

    Parameters
    ----------
    max_length : int
        The maximum number of peptides in the structure.
    n_results : int
        The number of results to keep.
    strict_proteins : bool = False
        If True, the search will exlude enzymes, transporters, inhibitors, etc.

    Returns
    -------
    results : list[str]
        The list of PDB IDs.

    """

    """q_keywords = [
        AttributeQuery(
            attribute="struct_keywords.pdbx_keywords",
            operator="contains_words",
            negation=True,
            value=f'"{word}"'
        ) for word in exclude_words
    ]

    q_title = [
        TextQuery(
            attribute="struct_title",
            operator="contains_words",
            negation=True,
            value=f'"{word}"'
        ) for word in exclude_words
    ]
    """
    # create terminals for each query

    q_type = (
        attrs.rcsb_entry_info.selected_polymer_entity_types == "Protein (only)"
    )
    q_pdb_comp = (
        attrs.pdbx_database_status.pdb_format_compatible == "Y"
    )
    q_length = attrs.rcsb_assembly_info.polymer_monomer_count <= max_length
    q_stricter = AttributeQuery(
        attribute="struct_keywords.pdbx_keywords",
        operator="contains_words",
        value="PROTEIN"
    )

    # combine using bitwise operators (&, |, ~, etc)
    query = q_type & q_pdb_comp & q_length
    
    if stricter_search:
        query = query & q_stricter

    random.seed(9)
    results = random.sample(list(query()), n_results)
    
    return results


def get_model_structure(
    attention: tuple[torch.Tensor, ...],
) -> tuple[
    int,
    int,
]:
    """
    Return the number of heads and the number of layers of ProtBert.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention from the model, either "raw" or cleared of the attention
        relative to tokens [CLS] and [SEP].

    Returns
    -------
    number_of_heads : int
        The number of heads of ProtBert.
    number_of_layers : int
        The number of layers of ProtBert.

    """
    layer_structure = attention[0].shape
    if len(layer_structure) == 4:  # i.e., in case of raw_attention
        number_of_heads = layer_structure[1]
    elif len(layer_structure) == 3:  # i.e., in case of "cleared" attention
        number_of_heads = layer_structure[0]
    number_of_layers = len(attention)

    return (
        number_of_heads,
        number_of_layers,
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
    model = BertModel.from_pretrained(
        model_name, output_attentions=True, attn_implementation="eager"
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_name, do_lower_case=False
    )

    return (
        model,
        tokenizer,
    )
