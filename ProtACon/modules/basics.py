"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella, R. Eliasy

This module defines:
    - the dictionaries for translating from multiple letter to single letter
      amino acid codes, and vice versa
    - the implementation of the AA-dataframe
    - the implementation of the CA_Atom class
    - funtions for extracting information from ProtBert and from PDB objects
    - a list with the twenty canonical amino acids

"""
from pathlib import Path
from typing import Any
import inspect
import random

from Bio.PDB.Structure import Structure
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import Flex, em, hw, ja
from rcsbsearchapi import rcsb_attributes as attrs
from rcsbsearchapi.search import AttributeQuery
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
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
        coords: list[float],
        hydropathy: float,
        volume: float,
        charge_density: float,
        rcharge_density: float,
        aa_ph: float,
    ):
        """
        Contructor of the class.

        Parameters
        ----------
        name : str
            The amino acid of the residue.
        idx : int
            The position of the residue along the chain.
        coords : list[float]
            The x-, y- and z- coordinates of the CA atom of the residue.
        hydropathy : float
            The hydropathy value by Kyte and Doolittle hydrophobicity(+)/
            hydrophilicity(-).
        volume : float
            The volume of the AA in cubic Angstroms (Å^3).
        charge_density : float
            The charge density of the AA in mA/Å^3, assuming the whole volume
            of the AA and a uniform distribution of the charge.
        rcharge_density : float
            The charge density of the AA in mA/Å^3, assuming only the volume
            occupied by the Rgroup and a uniform distribution of the charge.
        charge : float
            The charge of the AA in elementary charges.
        aa_ph : float
            The pH of the amino acid.

        """
        self.name = name
        self.idx = idx
        self.coords = coords
        self.hydropathy = hydropathy
        self.volume = volume
        self.charge_density = charge_density
        self.rcharge_density = rcharge_density
        self.charge = charge_density*volume
        self.aa_ph = aa_ph


def aromaticity_indicization(
    aa_name: str,
) -> int:
    """
    Parametrize the aromaticity of an amino acid.

    Parameters:
    -----------
    aa_name : str

    Returns:
    --------
    int
        1 if the amminoacid contain an aromatic ring,
        0 otherwise

    """
    aas_aromatics = 'YVF'
    if len(aa_name) != 1:
        raise ValueError('The amino acid must be a one-value-letter')
    else:
        if aa_name.upper() in aas_aromatics:
            return 1
        else:
            return 0


def assign_color_to(
    discrete_list_of: list,
    set_of_elements: set = None,
    case_sensitive: bool = False,
) -> dict | bool:
    """
    Consider the possibility to have a list of almost 10 different colors to
    map a dicrete set of values. Also for strings, to build a dictionary to
    convert each string into a color.

    Parameters:
    -----------
    discrete_list_of : list
        The list of discrete values to convert into colors.
    set_of_elements : set 
        The set of elements to consider for the color mapping. If not provided
        the set is built from the list.
    case_sensitive : bool
        The parameter to check if the mapping is case sensitive or not.
    Returns:
    --------
    color_dictionary : dict
        The dictionary with the map of the discrete values to the colors in the
        format :
        {element1 : 'red',
        element2 : 'blue',
        element3 : 'green',
        ...}
    """
    if set_of_elements == None:
        set_of_elements = set(discrete_list_of)
    if not case_sensitive:
        set_of_elements = set(
            [el.upper() for el in set_of_elements if isinstance(el, str)])
    if len(set_of_elements) > 10:
        return False
    color_list = [
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown',
        'black', 'grey'
    ]

    color_dictionary = dict(zip(set_of_elements, color_list))
    return color_dictionary


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

    r = requests.get(url, allow_redirects=True)

    with open(file_path, "wb") as file:
        file.write(r.content)

    return None


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
                        coords=atom.get_coord(),
                        hydropathy=dict_hydropathy_kyte_doolittle[
                            dict_3_to_1[residue.get_resname()]
                        ],
                        charge_density=dict_charge_density[
                            dict_3_to_1[residue.get_resname()]
                        ],
                        volume=dict_AA_volumes[
                            dict_3_to_1[residue.get_resname()]
                        ],
                        aa_ph=dict_AA_PH[dict_3_to_1[
                            residue.get_resname()]
                        ],
                        rcharge_density=dict_charge_density_Rgroups[
                            dict_3_to_1[residue.get_resname()]
                        ],
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
        value="PROTEIN"
    )

    # combine using bitwise operators (&, |, ~, etc)
    query = q_type & q_pdb_comp & q_min_length & q_max_length

    if stricter_search:
        query = query & q_stricter

    random.seed(9)
    results = random.sample(list(query()), n_results)

    return results


def get_AA_features_dataframe(
    CA_Atoms: tuple[CA_Atom, ...],
) -> pd.DataFrame:
    """
    Build a DataFrame containing the features of the amino acids.

    The DataFrame contains the following columns:
    - 'AA_Name': name of the amino acid
    - 'AA_Coords': x-, y- and z- coordinates of the CA atom of the amino acid
    - 'AA_Hydropathy': hydropathy value by Kyte and Doolittle
    hydrophobicity(+)/hydrophilicity(-)
    - 'AA_Volume': volume of the AA in cubic angstroms (Å^3)
    - 'AA_Charge_density': charge density of the AA in mA/Å^3,
    assuming the whole volume of the AA and an uniform distribution of the
    charge
    - 'AA_Rcharge_density': charge density of the AA in mA/Å^3, assuming only
    the volume occupied by the Rgroup and an uniform distribution of the charge
    - 'AA_Charge': charge of the AA in elementary charges
    - 'AA_localCharge' : the carge of a local triplet of amminoacids
    - 'AA_PH': pH of the amino acid
    - 'AA_isoPH': isoelectric point of the amino acid
    - 'AA_local_isoPH: the isoelectric point of central amminoacid considering
    a window_size of 3
    - 'AA_Hydrophilicity': hydrophilicity value of the amino acid
    - 'AA_Surface_accessibility': surface accessibility of the amino acid
    - 'AA_ja_transfer_energy_scale': transfer energy scale of the amino acid
    - 'AA_self_Flex': flexibility of the amino acid
    - 'AA_local_flexibility': local flexibility of the amino acid
    - 'AA_secondary_structure': index of the secondary structure of the amino
    acid
    - 'AA_aromaticity': aromaticity of the amino acid
    - 'AA_human_essentiality': essentiality of the amino acid
    - 'AA_web_group': group classification of the amino acid

    Parameters
    ----------
    CA_Atoms : tuple[CA_Atom, ...]
        the tuple of residual objs collected in CA_Atom objs

    Returns
    -------
    AA_features_dataframe : pd.DataFrame

    """
    protein_sequence_no_space = ''.join(AA.name for AA in CA_Atoms)
    flexibilities = local_flexibility(protein_sequence_no_space)
    local_isoPH = local_iso_PH(CA_Atoms=CA_Atoms, handle_border='same')
    local_charges = local_charge(CA_Atoms=CA_Atoms, handle_border='same')

    data = {  # dictionary to build the DataFrame
        'AA_Name': [AA.name for AA in CA_Atoms],
        'AA_Coords': [AA.coords for AA in CA_Atoms],
        'AA_Hydropathy': [AA.hydropathy for AA in CA_Atoms],
        'AA_Volume': [AA.volume for AA in CA_Atoms],
        'AA_Charge_Density': [AA.charge_density for AA in CA_Atoms],
        'AA_Rcharge_density': [AA.rcharge_density for AA in CA_Atoms],
        'AA_Charge': [AA.charge for AA in CA_Atoms],
        'AA_local_Charge': [charge for charge in local_charges],
        'AA_PH': [AA.aa_ph for AA in CA_Atoms],
        'AA_isoPH': [IsoelectricPoint(AA.name).pi() for AA in CA_Atoms],
        'AA_local_isoPH': [localPH for localPH in local_isoPH],
        'AA_Hydrophilicity': [hw[AA.name] for AA in CA_Atoms],
        'AA_Surface_accessibility': [em[AA.name] for AA in CA_Atoms],
        'AA_ja_transfer_energy_scale': [ja[AA.name] for AA in CA_Atoms],
        'AA_self_Flex': [Flex[AA.name] for AA in CA_Atoms],
        'AA_local_flexibility': [AA_flex for AA_flex in flexibilities],
        'AA_secondary_structure': [
            secondary_structure_index(AA.name) for AA in CA_Atoms
        ],
        'AA_aromaticity': [
            aromaticity_indicization(AA.name) for AA in CA_Atoms
        ],
        'AA_human_essentiality': [
            human_essentiality(AA.name) for AA in CA_Atoms
        ],
        'AA_web_group': [
            web_group_classification(AA.name) for AA in CA_Atoms
        ]
    }

    AA_features_dataframe = pd.DataFrame(
        data, index=[AA.idx for AA in CA_Atoms])

    return AA_features_dataframe


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
    CA_Atoms: tuple[CA_Atom, ...],
) -> str:
    """
    Return a string of the residues in a format suitable for tokenization.

    Take the name attribute of the CA_Atom objects in the tuple, translate it
    from multiple letter to single letter amino acid codes and append them to a
    single string, ready to be tokenized.

    Parameters
    ----------
    CA_Atoms : tuple[CA_Atom, ...]

    Returns
    -------
    sequence : str
        The sequence of residues.

    """
    sequence = ""
    for atom in CA_Atoms:
        sequence = sequence + atom.name + " "

    return sequence


def get_var_name(
    variable: Any
) -> str:
    """
    Get the name of the variable as a string.

    Parameters
    ----------
    variable : Any
        The variable to get the name from.

    Returns
    -------
    str
        The name of the variable.

    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_names = [name for name, val in callers_local_vars if val is variable]

    return str(var_names[0])


def human_essentiality(
    aa_name: str,
) -> float:
    """
    Parametrize the essentiality of the amino acid from the human POV.

    Parameters:
    -----------
    aa_name : str

    Returns:
    --------
    float
        The essentiality of the amino acid:
        - (-1) if not essential
        - (0) if conditionally essential
        - (1) if essential

    """
    is_essential = 'VLIRFTMKWH'
    is_conditional = 'YRCQGP'
    is_not_essential = 'NSADE'
    if len(aa_name) != 1:
        raise ValueError('The amino acid must be a one-value-letter')
    else:
        aa_name = aa_name.upper()
        if aa_name in is_essential:
            return 1.0
        elif aa_name in is_conditional:
            return 0.0
        elif aa_name in is_not_essential:
            return -1.0
        else:
            return np.nan


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


def local_charge(
    CA_Atoms: tuple[CA_Atom, ...],
    handle_border: str = "same",
) -> tuple[float, ...]:
    """
    Return the sums of the absolute charges in the triplet of AAs.

    Parameters:
    -----------
    CA_Atoms: tuple[CA_Atom, ...]
        The residues to take the subsequence from.
    handle_border: str, default='same'
        The parameter to choose the way to handle border; it can be:
        - 'same': extend the original chain of one slot for each end, with the
        same final aa.
        - 'mirror': mirror the next aa in the sequence from the other end.
        - 'couple': consider a couple made only of the first and second or the
        last and last minus one.

    Returns:
    --------
    tuple[float, ...] 
        The absolute charge of the triplet of amino acids along the chain.

    """
    protein_sequence = ''.join([AA.name for AA in CA_Atoms])
    protein_sequence_ns = str(protein_sequence.replace(' ', '')).upper()
    win_size = 3
    summed_charges = []
    initial = protein_sequence_ns[0]
    finale = protein_sequence_ns[-1]
    second = protein_sequence_ns[1]
    penultimate = protein_sequence_ns[-2]
    if handle_border.lower() == 'same':
        protein_sequence_ns = [initial] + list(protein_sequence_ns) + [finale]
    elif handle_border.lower() == 'mirror':
        protein_sequence_ns = [second] + \
            list(protein_sequence_ns) + [penultimate]

    for i in range(len(protein_sequence_ns) - win_size + 1):
        subsequence = list(protein_sequence_ns[i: i + win_size])
        summed_charges.append(
            sum([abs(dict_AA_charge[aa]) for aa in list(subsequence)]))
    if handle_border == 'couple':
        first_calculate = sum(
            [abs(dict_AA_charge[aa]) for aa in protein_sequence[:2]])
        last_calculate = sum(
            [abs(dict_AA_charge[aa]) for aa in protein_sequence[-2:]])
        summed_charges = [first_calculate] + summed_charges + [last_calculate]

    return tuple(summed_charges)


def local_flexibility(
    protein_sequence: tuple[CA_Atom, ...] | str,
) -> tuple[float, ...]:  # return the specific calcula for a window of size 9
    """
    Since Biopython doesn't solve the issue on the flexibility, we reintroduce
    the method from scratches. We handle the borders putting them to 0.

    Parameters
    ----------
    protein_sequence : str
        The sequence of residues
    protein_sequence : tuple[CA_Atom, ...]
        the list of CA_Atoms in get from residues

    Returns
    -------
    flexibility : tuple[float, ...]
        The tuple containing the flexibility of the residues in the sequence.

    """
    if isinstance(protein_sequence, tuple):
        protein = ''.join([AA.name for AA in protein_sequence])
    else:
        protein = protein_sequence
    protein_sequence_ns = str(protein.replace(' ', '')).upper()
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


def local_iso_PH(
    CA_Atoms: tuple[CA_Atom, ...],
    handle_border: str = 'couple',
) -> tuple[float, ...]:
    """
    Perform a computation on a peptide chain, introducing the isoelectric
    point of a triplet of amminoacids since the mean distance between amino
    acids in sequence is about half the range of interaction of 7 Angstroms.

    Parameters:
    -----------
    CA_Atoms: tuple[CA_Atom, ...]
        The residues to take the subsequence from.
    handle_border: str
        The parameter to choose the way to handle border; it can be: 
        - 'zero': set the borders to 0.0.
        - 'same': extend the original chain of one slot for each end, with the
        same final aa.
        - 'mirror': mirror the next aa in the sequence from the other end.
        - 'couple': consider a couple made only of the first and second or the
        last and last minus one.

    Returns:
    --------
    tuple[float, ...] 
        The isoelectric points of the triplet of residues along the chain.

    """

    res_chain_ns = ''.join([AA.name for AA in CA_Atoms])
    res_chain = [str(el) for el in res_chain_ns]
    win_size = 3
    iso_points = []
    initial = res_chain_ns[0]
    finale = res_chain_ns[-1]
    second = res_chain_ns[1]
    penultimate = res_chain_ns[-2]
    if handle_border.lower() == 'same':
        res_chain_ns = [initial] + [res_chain_ns] + [finale]
    elif handle_border.lower() == 'mirror':
        res_chain_ns = [second] + [res_chain_ns] + [penultimate]

    for i in range(len(res_chain_ns) - win_size + 1):
        subsequence = res_chain_ns[i: i + win_size]
        iso_points.append(IsoelectricPoint(str(subsequence)).pi())
    if handle_border == 'zero':
        iso_points = [0.0] + iso_points + [0.0]
    elif handle_border.lower() == 'couple':
        first_calculate = IsoelectricPoint(res_chain[:2])
        last_calculate = IsoelectricPoint(res_chain[-2:])
        iso_points = [first_calculate.pi()]+iso_points+[last_calculate.pi()]

    return tuple(iso_points)


def protein_reference_point(
    CA_Atoms: tuple[CA_Atom, ...],
) -> dict:
    """
    Calculate the reference point of the specific protein.

    Get the information from ProteinAnalysis of BioPython libs.

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
    protein_sequence = ''.join([AA.name for AA in CA_Atoms])
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


def secondary_structure_index(
    aa_name: str,
) -> int:
    """Return the index of the secondary structure of the amino acid."""

    is_helix = 'VIYFWL'
    is_turn = 'NPGS'
    is_sheet = 'EMAL'
    if len(aa_name) != 1:
        raise ValueError('The amino acid must be a one-value-letter')
    else:
        aa_name = aa_name.upper()
        if aa_name in is_helix:
            return 1
        elif aa_name in is_turn:
            return 2
        elif aa_name in is_sheet:
            return 3
        else:
            return 0  # for undefined structure


def web_group_classification(
    aa_name: str,
) -> int:
    """
    Parametrization of the classification of amino acids following the web
    literature: https://chimicamo.org/biochimica/gli-amminoacidi/.

    Parameters:
    -----------
    aa_name : str

    Returns:
    --------
    int
        The classification of the amino acid.

    """
    web_groups = {
        'A': 'G1',
        'L': 'G1',
        'I': 'G1',
        'V': 'G1',
        'P': 'G1',
        'M': 'G1',
        'F': 'G1',
        'W': 'G1',
        'S': 'G2',
        'T': 'G2',
        'Y': 'G2',
        'N': 'G2',
        'Q': 'G2',
        'C': 'G2',
        'G': 'G2',
        'K': 'G3',
        'H': 'G3',
        'R': 'G3',
        'D': 'G4',
        'E': 'G4',
    }
    if len(aa_name) != 1:
        raise ValueError('The amino acid must be a one-value-letter')
    else:
        if web_groups[aa_name] == 'G1':
            return 1
        elif web_groups[aa_name] == 'G2':
            return 2
        elif web_groups[aa_name] == 'G3':
            return 3
        elif web_groups[aa_name] == 'G4':
            return 4
    pass
