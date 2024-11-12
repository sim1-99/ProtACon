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

from transformers import BertModel, BertTokenizer
import pytest
import torch

from ProtACon.modules.attention import clean_attention
from ProtACon.modules.basics import (
    download_pdb,
    extract_CA_atoms,
    get_model_structure,
    get_sequence_to_tokenize,
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
def encoded_input(sequence):
    """List of int got from the encoding of sequence."""
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        clean_up_tokenization_spaces=True,
    )
    encoded_input = tokenizer.encode(sequence, return_tensors='pt')
    return encoded_input


@pytest.fixture(scope="module")
def tokens(encoded_input, tokenizer):
    """List of tokens from the encoded input, exluding [CLS] and [SEP]."""
    raw_tokens = tokenizer.convert_ids_to_tokens(encoded_input[0])
    tokens = raw_tokens[1:-1]
    return tokens


@pytest.fixture(scope="session")
def output(encoded_input):
    """Output from ProtBert."""
    model = BertModel.from_pretrained(
        model_name,
        output_attentions=True,
        attn_implementation="eager",
    )
    with torch.no_grad():
        output = model(encoded_input)
    return output


@pytest.fixture(scope="module")
def raw_attention(output):
    """Attention from ProtBert, including non-amino acid tokens."""
    raw_attention = output[-1]
    return raw_attention


@pytest.fixture(scope="session")
def attention(output):
    """Attention from ProtBert, cleaned of non-amino acid tokens."""
    attention = clean_attention(output[-1])
    return attention


@pytest.fixture(scope="session")
def model_structure(attention):
    """Tuple with the number of heads and layers of ProtBert."""
    n_heads, n_layers = get_model_structure(attention)
    return n_heads, n_layers
