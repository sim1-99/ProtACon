"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-07

Collection of pytest fixtures.

"""
from pathlib import Path
import os
import warnings

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.PDBParser import PDBParser

import pytest

from ProtACon.modules.basics import (
    download_pdb,
    extract_CA_atoms,
    get_model_structure,
    get_sequence_to_tokenize
)


@pytest.fixture(scope="session")
def data_path():
    """Path to the directory containing the PDB files."""
    return Path(__file__).resolve().parent/"test_data"


@pytest.fixture(scope="session", params=["1HPV", "2ONX"])
def chain_ID(request):
    """The PDB ID of a peptide chain."""
    return request.param


@pytest.fixture(scope="session")
def structure(chain_ID, data_path):
    """Structure of a peptide chain."""
    download_pdb(chain_ID, data_path)
    pdb_path = data_path/f"pdb{chain_ID.lower()}.ent"

    with warnings.catch_warnings():
        # warn that the chain is discontinuous, this is not a problem though
        warnings.simplefilter('ignore', PDBConstructionWarning)
        structure = PDBParser().get_structure(chain_ID, pdb_path)

    yield structure
    # Teardown
    test_files = os.listdir(data_path)
    for item in test_files:
        if item.endswith(".ent"):
            os.remove(os.path.join(data_path, item))


@pytest.fixture(scope="session")
def CA_atoms(structure):
    """Tuple of the CA_Atom objects of a peptide chain."""
    return extract_CA_atoms(structure)


@pytest.fixture(scope="session")
def sequence(CA_atoms):
    """String with the amino acids of the residues in a peptide chain."""
    return get_sequence_to_tokenize(CA_atoms)


@pytest.fixture(scope="session")
def model_name():
    """Name of the ProtBert model."""
    return "Rostlab/prot_bert"


@pytest.fixture(scope="session")
def model(model_name):
    """Object of type transformers.BertModel."""
    model = BertModel.from_pretrained(
        model_name,
        output_attentions=True,
        attn_implementation="eager",
    )
    return model


@pytest.fixture(scope="session")
def tokenizer(model_name):
    """Object of type transformers.BertTokenizer."""
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        clean_up_tokenization_spaces=True,
    )
    return tokenizer


@pytest.fixture(scope="session")
def encoded_input(sequence, tokenizer):
    """List of int got from the encoding of sequence."""
    encoded_input = tokenizer.encode(sequence, return_tensors='pt')
    return encoded_input


@pytest.fixture(scope="session")
def raw_attention(encoded_input, model):
    """Attention extracted from ProtBert."""
    output = model(encoded_input)
    raw_attention = output[-1]
    return raw_attention


