#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miscellaneous.

This module contains:
    - the dictionaries for translating from multiple letter to single letter
      amino acid codes, and vice versa
    - dictionaries containing the information about the amino acids
    - the building of the AA-dataframe
    - the implementation of the CA_Atom class
    - funtions for extracting information from ProtBert and from PDB objects
"""

__author__ = 'Simone Chiarella, Renato Eliasy'
__email__ = 'simone.chiarella@studio.unibo.it, renato.eliasy@studio.unibo.it'

import logging
from Bio.SeqUtils.ProtParamData import Flex, kd, hw, em, ja, DIWV
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Structure import Structure
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd


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

dict_hydropathy_kyte_doolittle = {  # the same as kd of Bio.SeqUtils.ProtParamData
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2
}

dict_AA_charge = {
    "A": 0,   # (neutral)
    "R": 1,   # (positively charged)
    "N": 0,
    "D": -1,  # (negatively charged)
    "C": 0,
    "Q": 0,
    "E": -1,
    "G": 0,
    "H": 0,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 0,
    "T": 0,
    "W": 0,
    "Y": 0,
    "V": 0
}

# Values are approximate and in cubic angstroms (Å^3)
dict_AA_volumes = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0
}

dict_AA_PH = {
    'A': 6.01,
    'R': 10.76,
    'N': 5.41,
    'D': 2.77,
    'C': 5.07,
    'Q': 5.65,
    'E': 3.22,
    'G': 5.97,
    'H': 7.59,
    'I': 6.02,
    'L': 5.98,
    'K': 9.74,
    'M': 5.74,
    'F': 5.48,
    'P': 6.30,
    'S': 5.68,
    'T': 5.60,
    'W': 5.89,
    'Y': 5.66,
    'V': 5.96
}

dict_charge_density_Rgroups = {  # considering only the volume occupied by the Rgroup
    'A': 0.0,
    'R': 8.8261253309797,  # provided in mA/Å^3
    'N': 0.0,
    'D': -19.607843137254903,
    'C': 0.0,
    'Q': 0.0,
    'E': -12.771392081736908,
    'G': 0.0,
    'H': 0.0,
    'I': 0.0,
    'L': 0.0,
    'K': 9.216589861751151,
    'M': 0.0,
    'F': 0.0,
    'P': 0.0,
    'S': 0.0,
    'T': 0.0,
    'W': 0.0,
    'Y': 0.0,
    'V': 0.0
}

dict_charge_density = {  # considering the whole volume of the amino acid
    'A': 0.0,
    'R': 5.767,  # provided in mA/Å^3
    'N': 0.0,
    'D': -9.0009,
    'C': 0.0,
    'Q': 0.0,
    'E': -7.2254,
    'G': 0.0,
    'H': 0.0,
    'I': 0.0,
    'L': 0.0,
    'K': 5.9312,
    'M': 0.0,
    'F': 0.0,
    'P': 0.0,
    'S': 0.0,
    'T': 0.0,
    'W': 0.0,
    'Y': 0.0,
    'V': 0.0
}


class CA_Atom:
    """A class to represent CA atoms of amino acids."""

    def __init__(
        self,
        name: str,
        idx: int,
        coords: list[float],
        hydropathy: float,
        volume: float,
        charge_density: float,
        Rcharge_density: float,
        aa_ph: float
    ):
        """
        Contructor of the class.

        Parameters
        ----------
        name : str
            name of the amino acid
        idx : int
            position of the amino acid along the chain
        coords : list[float]
            x-, y- and z- coordinates of the CA atom of the amino acid
        hydropathy : float
            hydropathy value by Kyte and Doolittle hydrophobicity(+)/hydrophilicity(-)
        volume : float
            volume of the AA in cubic angstroms (Å^3)
        charge_density : float
            charge density of the AA in mA/Å^3, assuming the whole volume of the AA and an uniform distribution of the charge
        Rcharge_density : float
            charge density of the AA in mA/Å^3, assuming only the volume occupied by the Rgroup and an uniform distribution of the charge
        charge : float
            charge of the AA in elementary charges
        aa_ph : float
            pH of the amino acid
        """
        self.name = name
        self.idx = idx
        self.coords = coords
        self.hydropathy = hydropathy
        self.volume = volume
        self.charge_density = charge_density
        self.Rcharge_density = Rcharge_density
        self.charge = charge_density*volume
        self.aa_ph = aa_ph


def extract_CA_Atoms(
    structure: Structure
) -> tuple[CA_Atom, ...]:
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
                    coords=atom.get_coord(),
                    hydropathy=dict_hydropathy_kyte_doolittle[dict_3_to_1[residue.get_resname(  # change with Bio.SeqUtils.ProtParamData
                    )]],
                    charge_density=dict_charge_density[dict_3_to_1[residue.get_resname(
                    )]],
                    volume=dict_AA_volumes[dict_3_to_1[residue.get_resname()]],
                    # change with Bio.SeqUtils.ProtParamData iso ph
                    aa_ph=dict_AA_PH[dict_3_to_1[residue.get_resname()]],
                    Rcharge_density=dict_charge_density_Rgroups[dict_3_to_1[residue.get_resname()]])

                )
                break
            elif atom.get_name() == "CA":
                logging.warning(" Found and discarded ligand in position: "
                                f"{residue_idx}")
    CA_Atoms_tuple = tuple(CA_Atoms_list)

    return CA_Atoms_tuple


def local_flexibility(protein_sequence: str  # the protein sequence, possibly without space char
                      ) -> tuple[float, ...]:  # return the specific calcula for a window of size 9
    """
    Since the biopython repository doesn't solve the issue on the flexibility,
    we have to correct and reintroduce the method from scratches
    we handle the borders putting them to 0

    Parameters
    ----------
    protein_sequence : str
        sequence of amino acids

    Returns
    -------
    flexibility : tuple(float, ...)
        tuple containing the flexibility of the amino acids in the protein sequence
    """
    protein_sequence_ns = str(protein_sequence.replace(' ', '')).upper()
    flexibilities = Flex
    window_size = 9
    weights = [0.25, 0.4375, 0.625, 0.8125, 1]
    scores = []

    for i in range(len(protein_sequence_ns) - window_size + 1):
        subsequence = protein_sequence_ns[i: i + window_size]
        score = 0.0

        for j in range(window_size // 2):
            front = subsequence[j]
            back = subsequence[window_size - j - 1]
            score += (flexibilities[front] + flexibilities[back]) * weights[j]

        middle = subsequence[window_size // 2]
        score += flexibilities[middle]

        scores.append(score / 5.25)

    # since the first and last 4 are not computed in the score count:
    border_handle = [0, 0, 0, 0]
    AA_flexibilities = border_handle + scores + border_handle

    if len(AA_flexibilities) != len(protein_sequence_ns):
        raise Exception('something wrong in the calculus of flexibilities')
    else:
        return tuple(AA_flexibilities)


def secondary_structure_index(amminoacid_name: str) -> int:
    """
    Return the index of the secondary structure of the amminoacid
    """
    is_helix = 'VIYFWL'
    is_turn = 'NPGS'
    is_sheet = 'EMAL'
    if len(amminoacid_name) != 1:
        raise ValueError('The name of amminoacids must be a one-value-letter')
    else:
        if amminoacid_name in is_helix:
            return 1
        elif amminoacid_name in is_turn:
            return 2
        elif amminoacid_name in is_sheet:
            return 3
        else:
            return 0  # for undefined structure


def aromaticity_indicization(name_of_amminoacids: str
                             ) -> int:
    """
    Parametrization of the aromaticity of an amminoacids:

    Parameters:
    -----------
    name_of_amminoacids : str
        the name of the amminoacid

    Returns:
    --------
    int
        1 if the amminoacid contain an aromatic ring, 
        0 otherwise
    """
    aas_aromatics = 'YVF'
    if len(name_of_amminoacids) != 1:
        raise ValueError('The name of amminoacids must be a one-value-letter')
    else:
        if name_of_amminoacids.upper() in aas_aromatics:
            return 1
        else:
            return 0


def get_AA_features_dataframe(
    CA_Atoms: tuple[CA_Atom, ...]
) -> pd.DataFrame:
    """
    Build a DataFrame containing the features of the amino acids.

    The DataFrame contains the following columns:
    - 'AA_Name': name of the amino acid
    - 'AA_Coords': x-, y- and z- coordinates of the CA atom of the amino acid
    - 'AA_Hydropathy': hydropathy value by Kyte and Doolittle hydrophobicity(+)/hydrophilicity(-)
    - 'AA_Volume': volume of the AA in cubic angstroms (Å^3)
    - 'AA_Charge_density': charge density of the AA in mA/Å^3, assuming the whole volume of the AA and an uniform distribution of the charge
    - 'AA_Rcharge_density': charge density of the AA in mA/Å^3, assuming only the volume occupied by the Rgroup and an uniform distribution of the charge
    - 'AA_Charge': charge of the AA in elementary charges
    - 'AA_PH': pH of the amino acid
    - AA.idx: position of the amino acid along the chain is used as index of the DataFrame

    Parameters
    ----------
    CA_Atoms : tuple[CA_Atom, ...]

    Returns
    -------
    AA_features_dataframe : pd.DataFrame

    """
    protein_sequence_no_space = ''.join(AA.name for AA in CA_Atoms)
    flexibilities = local_flexibility(protein_sequence_no_space)

    data = {                                        # dictionary to build the DataFrame
        'AA_Name': [AA.name for AA in CA_Atoms],
        'AA_Coords': [AA.coords for AA in CA_Atoms],
        'AA_Hydropathy': [AA.hydropathy for AA in CA_Atoms],
        'AA_Volume': [AA.volume for AA in CA_Atoms],
        'AA_Charge_Density': [AA.charge_density for AA in CA_Atoms],
        'AA_Rcharge_density': [AA.Rcharge_density for AA in CA_Atoms],
        'AA_Charge': [AA.charge for AA in CA_Atoms],
        'AA_PH': [AA.aa_ph for AA in CA_Atoms],
        'AA_isoPH': [IsoelectricPoint(AA.name).pi() for AA in CA_Atoms],
        'AA_Hydrophilicity': [hw[AA.name] for AA in CA_Atoms],
        'AA_Surface_accessibility': [em[AA.name] for AA in CA_Atoms],
        'AA_ja_transfer_energy_scale': [ja[AA.name] for AA in CA_Atoms],
        'AA_self_Flex': [Flex[AA.name] for AA in CA_Atoms],
        'AA_local_flexibility': [AA_flex for AA_flex in flexibilities],
        'AA_secondary_structure': [secondary_structure_index(AA.name) for AA in CA_Atoms],
        'AA_aromaticity': [aromaticity_indicization(AA.name) for AA in CA_Atoms]

    }

    AA_features_dataframe = pd.DataFrame(
        data, index=[AA.idx for AA in CA_Atoms])

    return AA_features_dataframe


def protein_reference_point(protein_sequence: str
                            ) -> dict:
    """
    Calculate the reference point of the specific protein, get information from ProteinAnalysis of BioPython libs
    Parameters
    ----------
    protein_sequence : str
        sequence of amino acids

    Returns
    -------
    reference_point : dict
        dictionary containing the reference point of the protein:
        - 'molecular_weight': float
        - 'aromaticity': float
        - 'instability_index': float
        - 'flexibility': float
        - 'isoelectric_point': float
        - 'mono isotopic' : bool
        - 'gravy' : float
        - 'secondary_structure_inclination' : dict of floats


    """

    protein_sequence_ns = str(protein_sequence.replace(' ', ''))
    protein = ProteinAnalysis(protein_sequence_ns.upper())
    reference_points = {
        'molecular_weight': protein.molecular_weight(),
        'aromaticity': protein.aromaticity(),
        'instability_index': protein.instability_index(),
        'flexibility': protein.flexibility(),
        'isoelectric_point': protein.isoelectric_point(),
        'mono isotopic': protein.monoisotopic(),
        'gravy': protein.gravy(),
        'secondary_structure_inclination': {
            'Helix_propensity': protein.secondary_structure_fraction()[0],
            'Turn_propensity': protein.secondary_structure_fraction()[1],
            'Sheet_propensity': protein.secondary_structure_fraction()[2]
        }

    }
    return reference_points


def get_model_structure(
    raw_attention: tuple[torch.Tensor, ...]
) -> tuple[
    int,
    int
]:
    """
    Return the number of heads and the number of layers of ProtBert.

    Parameters
    ----------
    raw_attention : tuple[torch.Tensor, ...]
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

    return (
        get_model_structure.number_of_heads,
        get_model_structure.number_of_layers
    )


def get_sequence_to_tokenize(
    CA_Atoms: tuple[CA_Atom, ...]
) -> str:
    """
    Return a string of amino acids in a format suitable for tokenization.

    The function takes the name attribute of the CA_Atom objects in the tuple,
    translate them from multiple letter to single letter amino acid codes and
    append them to a single string, ready to be tokenized.

    Parameters
    ----------
    CA_Atoms : tuple[CA_Atom, ...]

    Returns
    -------
    sequence : str
        sequence of amino acids

    """
    sequence = ""
    for atom in CA_Atoms:
        sequence = sequence + atom.name + " "

    return sequence


def get_types_of_amino_acids(
    tokens: list[str]
) -> list[str]:
    """
    Return a list with the types of the residues present in the peptide chain.

    Parameters
    ----------
    tokens : list[str]
        contains strings which are the tokens used by the model, cleared of the
        tokens [CLS] and [SEP]

    Returns
    -------
    types_of_amino_acids : list[str]
        contains strings with single letter amino acid codes of the amino acid
        types in the peptide chain

    """
    types_of_amino_acids = list(dict.fromkeys(tokens))

    return types_of_amino_acids


def load_model(
    model_name: str
) -> tuple[
    BertModel,
    BertTokenizer
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
