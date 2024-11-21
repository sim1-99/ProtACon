"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

This module defines:
    - the dictionaries for translating from multiple letter to single letter
      amino acid codes, and vice versa
    - a list with the twenty canonical amino acids
    - the implementation of the CA_Atom class
    - functions for extracting information from ProtBert and from PDB objects
    - a function to fetch PDB entries according to some queries
    - a function to download .ent (i.e., pdb) files
    - a function for normalizing numpy arrays

"""
from pathlib import Path
import random

from Bio.PDB.Structure import Structure
from rcsbsearchapi import rcsb_attributes as attrs
from rcsbsearchapi.search import AttributeQuery
from transformers import BertModel, BertTokenizer
import numpy as np
import requests
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

log = Logger("mylog").get_logger()


class CA_Atom:
    """A class to represent the CA atoms of the residues."""

    def __init__(
        self,
        name: str,
        idx: int,
        coords: np.ndarray,
    ):
        """
        Contructor of the class.

        Parameters
        ----------
        name : str
            The amino acid of the residue.
        idx : int
            The position of the residue along the chain.
        coords : np.ndarray
            The x-, y- and z- coordinates of the CA atom of the residue.

        """
        self.name = name
        self.idx = idx
        self.coords = coords


def download_pdb(
    pdb_code: str,
    pdb_dir: Path,
    download_url: str = "https://files.rcsb.org/download/",
) -> None:
    """
    Downloads a PDB file from the Internet and saves it in a data directory.

    I am creating a function by myself because the function
    Bio.PDB.PDBList.retrieve_pdb_file relies on the PDB FTP service, that may
    have issues with firewalls.

    Parameters
    ----------
    pdb_code : str
        The PDB code of the chain.
    pdb_dir : Path
        The directory where to save the PDB file.
    download_url : str, default="https://files.rcsb.org/download/"
        The URL to download the PDB file from.

    Returns
    -------
    None

    """
    fn_in = pdb_code + ".pdb"
    fn_out = "pdb" + pdb_code.lower() + ".ent"  # adapt to preprocess.py
    file_path = pdb_dir/fn_out

    if file_path.is_file():
        log.logger.warning(
            f"A file with the same path already exists: {file_path}\n"
            "The pdb file will not be saved."
        )
        return None

    url = download_url + fn_in

    r = requests.get(url)

    with open(file_path, "wb") as file:
        file.write(r.content)

    return None


def extract_CA_atoms(
    structure: Structure,
) -> tuple[CA_Atom, ...]:
    """
    Extract the CA atoms from the a peptide chain.

    Given a Structure, take the first chain. Then, look for the CA atoms (main
    carbon atom of each residue) and for each of them get:
        - the amino acids of the residue
        - the position index in the chain
        - the coordinates of the CA atom
    Finally, initialize a CA_Atom object with this information and append it to
    a list. Repeat for all the CA atoms in the chain and return the list as a
    tuple.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The object containing information about each atom of the peptide chain.

    Returns
    -------
    tuple[CA_Atom, ...]

    """
    chains = structure[0].get_list()
    chain = chains[0]
    log.logger.info(f"Taking chain: [purple]{chain.get_id()}")

    residues = chain.get_list()
    CA_atoms = []

    for residue_idx, residue in enumerate(residues):
        for atom in residue:
            if (
                atom.get_name() == "CA" and
                residue.get_resname() in dict_3_to_1
            ):
                CA_atoms.append(
                    CA_Atom(
                        name=dict_3_to_1[residue.get_resname()],
                        idx=residue_idx,
                        coords=atom.get_coord(),
                    )
                )
                break
            elif atom.get_name() == "CA":
                log.logger.info(
                    f"Found and discarded ligand in position: {residue_idx}"
                )

    return tuple(CA_atoms)


def fetch_pdb_entries(
    min_length: int,
    max_length: int,
    n_results: int,
    stricter_search: bool = False,
) -> list[str]:
    """
    Fetch PDB entries.

    The query consists in returning proteins with a minimum and a maximum
    number of peptides in the structure. Keep only the number of results
    specified by n_results.

    Parameters
    ----------
    min_length : int
        The minimum number of peptides in the structure.
    max_length : int
        The maximum number of peptides in the structure.
    n_results : int
        The number of results to keep.
    strict_proteins : bool, default=False
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
    q_min_length = attrs.rcsb_assembly_info.polymer_monomer_count >= min_length
    q_max_length = attrs.rcsb_assembly_info.polymer_monomer_count <= max_length
    q_stricter = AttributeQuery(
        attribute="struct_keywords.pdbx_keywords",
        operator="contains_words",
        value="PROTEIN",
    )

    # combine using bitwise operators (&, |, ~, etc)
    query = q_type & q_pdb_comp & q_min_length & q_max_length

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
    n_heads : int
        The number of heads of ProtBert.
    n_layers : int
        The number of layers of ProtBert.

    """
    layer_structure = attention[0].shape
    if len(layer_structure) == 4:  # i.e., in case of raw_attention
        n_heads = layer_structure[1]
    elif len(layer_structure) == 3:  # i.e., in case of "cleared" attention
        n_heads = layer_structure[0]
    n_layers = len(attention)

    return (
        n_heads,
        n_layers,
    )


def get_sequence_to_tokenize(
    CA_atoms: tuple[CA_Atom, ...],
) -> str:
    """
    Return a string of the residues in a format suitable for tokenization.

    Take the name attribute of the CA_Atom objects in the tuple and put them,
    separated with spaces, in a single string ready to be tokenized.

    Parameters
    ----------
    CA_atoms : tuple[CA_Atom, ...]

    Returns
    -------
    sequence : str
        The sequence of residues.

    """
    sequence = " ".join(atom.name for atom in CA_atoms)
    return sequence


def load_Bert(
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
        model_name,
        output_attentions=True,
        attn_implementation="eager",
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        clean_up_tokenization_spaces=True,
    )

    return (
        model,
        tokenizer,
    )


def normalize_array(
    array: np.ndarray,
) -> np.ndarray:
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
