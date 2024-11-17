"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-14

Fixtures simulating real objects used within the pipeline, for testing the
scripts.

"""
from transformers import BertModel, BertTokenizer
import pytest
import torch

from ProtACon.modules.attention import (
    clean_attention,
    threshold_attention,
)
from ProtACon.modules.basics import (
    extract_CA_atoms,
    get_model_structure,
    get_sequence_to_tokenize,
)


@pytest.fixture(scope="session", params=["1HPV", "2ONX"])
def chain_ID(request):
    """The PDB ID of a peptide chain."""
    return request.param


@pytest.fixture(scope="session")
def CA_atoms(structure):
    """Tuple of the CA_Atom objects of a peptide chain."""
    return extract_CA_atoms(structure)


@pytest.fixture(scope="session")
def sequence(CA_atoms):
    """String with the amino acids of the residues in a peptide chain."""
    return get_sequence_to_tokenize(CA_atoms)


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


@pytest.fixture(scope="module")
def tokens(encoded_input, tokenizer):
    """List of tokens from the encoded input, exluding [CLS] and [SEP]."""
    raw_tokens = tokenizer.convert_ids_to_tokens(encoded_input[0])
    tokens = raw_tokens[1:-1]
    return tokens


@pytest.fixture(scope="session")
def output(encoded_input, model_name):
    """Output from ProtBert."""
    model = BertModel.from_pretrained(
        model_name,
        output_attentions=True,
        attn_implementation="eager",
    )
    with torch.no_grad():
        output = model(encoded_input)
    return output


@pytest.fixture(scope="session")
def model_structure(attention):
    """Tuple with the number of heads and layers of ProtBert."""
    n_heads, n_layers = get_model_structure(attention)
    return n_heads, n_layers


@pytest.fixture(scope="session")
def attention(output):
    """Tuple of tensors storing attention, cleaned of non-amino acid tokens."""
    attention = clean_attention(output[-1])
    return attention


@pytest.fixture(scope="module")
def raw_attention(output):
    """Tuple of tensors storing attention, including non-amino acid tokens."""
    raw_attention = output[-1]
    return raw_attention


@pytest.fixture(scope="module")
def thresholded_attention(output):
    """Tuple of tensors with all values below att_cutoff set to zero."""
    att_cutoff = 0.5
    thresholded_attention = threshold_attention(output[-1], att_cutoff)
    return thresholded_attention
