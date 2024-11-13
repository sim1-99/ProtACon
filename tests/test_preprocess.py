"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-07

Test suite for preprocess.py.

"""
import numpy as np
import pytest
import torch

from ProtACon.modules.basics import (
    CA_Atom,
    all_amino_acids,
)
from ProtACon.preprocess import main


pytestmark = pytest.mark.preprocess


# Tests
@pytest.mark.extract_CA_atoms
def test_CA_atoms_is_tuple_of_CA_Atom(CA_atoms):
    """
    Test that extract_CA_atoms() returns a tuple of CA_Atom objects.

    GIVEN: a Bio.PDB.Structure object
    WHEN: I call extract_CA_atoms()
    THEN: the function returns a tuple of CA_Atom objects

    """
    assert isinstance(CA_atoms, tuple)
    assert all(isinstance(atom, CA_Atom) for atom in CA_atoms)


@pytest.mark.extract_CA_atoms
def test_CA_atoms_data(CA_atoms):
    """
    Test that the CA_Atom objects in the tuple from extract_CA_atoms() have
    correct attributes.

    GIVEN: a Bio.PDB.Structure object
    WHEN: I call extract_CA_atoms()
    THEN: the CA_Atom objects in the tuple returned have correct attributes

    """
    ''' every amino acid is in the list of the twenty canonical amino acids;
    this also tests that no ligands are present in the CA_Atom objects
    '''
    assert all(atom.name in all_amino_acids for atom in CA_atoms)
    # the index of the amino acid is a non-negative integer
    assert all(atom.idx >= 0 for atom in CA_atoms)
    # the coordinates of the amino acid are three floats
    assert all(len(atom.coords) == 3 for atom in CA_atoms)
    assert all(
        isinstance(coord, np.float32)
        for atom in CA_atoms for coord in atom.coords
    )


@pytest.mark.get_sequence_to_tokenize
def test_sequence_is_string(sequence):
    """
    Test that get_sequence_to_tokenize() returns a string.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the function returns a string

    """
    assert isinstance(sequence, str)


@pytest.mark.get_sequence_to_tokenize
def test_chars_are_alpha_and_spaces(sequence):
    """
    Test that the sequence from get_sequence_to_tokenize() is composed of
    alphabetic characters and spaces.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the string returned is composed of alphabetic characters and spaces

    """
    assert all(char.isalpha() or char == " " for char in sequence)


@pytest.mark.get_sequence_to_tokenize
def test_alpha_chars_are_canonical_amino_acids(sequence):
    """
    Test that the alphabetic characters in the sequence from
    get_sequence_to_tokenize() represent the twenty canonical amino acids.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the alphabetic characters in the string returned represent the twenty
        canonical amino acids

    """
    assert all(char in all_amino_acids for char in sequence if char.isalpha())


@pytest.mark.get_sequence_to_tokenize
def test_spaces_between_chars(sequence):
    """
    Test that the alphabetic characters in the sequence from
    get_sequence_to_tokenize() are separated with spaces.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the alphabetic characters in the string returned are separated with
        spaces

    """
    assert all(char.isalpha() for i, char in enumerate(sequence) if i % 2 == 0)


@pytest.mark.get_sequence_to_tokenize
def test_sequence_length(CA_atoms, sequence):
    """
    Test that the sequence from get_sequence_to_tokenize() has the right
    length.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the string returned has the right length

    """
    assert len(sequence) == len(CA_atoms)*2-1  # consider spaces between chars


@pytest.mark.tokens
def test_tokens_are_the_same_as_sequence(sequence, tokens):
    """
    Test that the tokens extracted from the encoded input are the same as the
    amino acids in the sequence of residues, from which the encoded input
    itself is derived.

    GIVEN: the sequence of amino acids of the residues and the tokens extracted
        from the encoded input
    THEN: the tokens are the same as the sequence

    """
    assert all(token == sequence[i*2] for i, token in enumerate(tokens))


@pytest.mark.raw_attention
def test_raw_attention_is_tuple_of_tensors(raw_attention):
    """
    Test that the attention extracted from ProtBert is a tuple of torch.Tensor.

    GIVEN: raw_attention extracted from ProtBert
    THEN: raw_attention is a tuple of torch.Tensor

    """
    assert isinstance(raw_attention, tuple)
    assert all(isinstance(tensor, torch.Tensor) for tensor in raw_attention)


@pytest.mark.raw_attention
def test_raw_attention_n_layers(raw_attention):
    """
    Test that the tensors in the tuple storing the attention are 30 -- i.e.,
    the number of layers of ProtBert.

    GIVEN: raw_attention extracted from ProtBert
    THEN: raw_attention has length 30

    """
    assert len(raw_attention) == 30


@pytest.mark.raw_attention
def test_raw_attention_batch_size(raw_attention):
    """
    Test that the tensors in the tuple storing the attention have the first
    dimension equal to the batch size -- i.e., 1.

    GIVEN: raw_attention extracted from ProtBert
    THEN: the attention tensors have the first dimension equal to 1

    """
    assert all(tensor.shape[0] == 1 for tensor in raw_attention)


@pytest.mark.raw_attention
def test_raw_attention_n_heads(raw_attention):
    """
    Test that the tensors in the tuple storing the attention have the second
    dimension equal to the number of heads -- i.e., 16.

    GIVEN: raw_attention extracted from ProtBert
    THEN: the attention tensors have the second dimension equal to 16

    """
    assert all(tensor.shape[1] == 16 for tensor in raw_attention)


@pytest.mark.raw_attention
def test_raw_attention_seq_len(CA_atoms, raw_attention):
    """
    Test that the tensors in the tuple storing the attention have the third and
    the fourth dimensions equal to the sequence length -- i.e., the length of
    the peptide chain -- plus two, because of the tokens [CLS] and [SEP].

    GIVEN: raw_attention extracted from ProtBert
    THEN: the attention tensors have the third and the fourth dimensions equal
        to the the length of the peptide chain CA_atoms plus two

    """
    assert all(tensor.shape[2] == len(CA_atoms)+2 for tensor in raw_attention)
    assert all(tensor.shape[3] == len(CA_atoms)+2 for tensor in raw_attention)


@pytest.mark.raw_attention
def test_each_tensor_row_sums_1(raw_attention):
    """
    Test that the sum of the values in each row of each attention matrix is
    equal to 1.

    GIVEN: raw_attention extracted from ProtBert
    THEN: the sum of the values in each row of each attention matrix is equal
        to 1

    """
    assert all(
        torch.sum(raw_attention[i][0, j], 1)[k] == pytest.approx(1.)
        for i in range(len(raw_attention))
        for j in range(raw_attention[i].shape[1])
        for k in range(raw_attention[i].shape[2])
    )


@pytest.mark.raw_attention
def test_each_tensor_sums_seq_len(CA_atoms, raw_attention):
    """
    Test that the sum of the values in each attention matrix is equal to the
    length of the peptide chain plus two (tokens [CLS] and [SEP]).

    GIVEN: raw_attention extracted from ProtBert
    THEN: the sum of the values in each attention matrix is equal to the length
        of the peptide chain plus two

    """
    assert all(
        torch.sum(raw_attention[i][0, j]) == pytest.approx(len(CA_atoms)+2)
        for i in range(len(raw_attention))
        for j in range(raw_attention[i].shape[1])
    )


@pytest.mark.clean_attention
def test_clean_attention_returns_tuple_of_tensors(attention):
    """
    Test that clean_attention() returns a tuple of torch.Tensor.

    GIVEN: raw_attention from ProtBert
    WHEN: I call clean_attention()
    THEN: the function returns a tuple of torch.Tensor

    """
    assert isinstance(attention, tuple)
    assert all(isinstance(tensor, torch.Tensor) for tensor in attention)


@pytest.mark.clean_attention
def test_cleaned_attention_len(attention, raw_attention):
    """
    Test that the tuple returned by clean_attention() has the same length as
    the tuple raw_attention.

    GIVEN: raw_attention from ProtBert
    WHEN: I call clean_attention()
    THEN: the tuple returned has the same length as raw_attention

    """
    assert len(attention) == len(raw_attention)


@pytest.mark.clean_attention
def test_cleaned_attention_shape(attention, raw_attention):
    """
    Having the tensors in raw_attention shape (batch_size, n_heads, seq_len+2,
    seq_len+2), test that the tensors in attention have shape (n_heads,
    seq_len, seq_len).

    GIVEN: tensors in raw_attention with shape (batch_size, n_heads, seq_len+2,
        seq_len+2)
    WHEN: I call clean_attention()
    THEN: the tensors returned have shape (n_heads, seq_len, seq_len)

    """
    assert all(
        t1.shape[0] == t2.shape[1]
        for t1, t2 in zip(attention, raw_attention)
    )
    assert all(
        t1.shape[1] == t2.shape[2]-2
        for t1, t2 in zip(attention, raw_attention)
    )
    assert all(
        t1.shape[2] == t2.shape[3]-2
        for t1, t2 in zip(attention, raw_attention)
    )


@pytest.mark.clean_attention
def test_cleaned_attention_sums(attention, raw_attention):


@pytest.mark.get_model_structure
def test_get_model_structure_returns_ints(model_structure):
    """
    Test that get_model_structure() returns two integers.

    GIVEN: attention from ProtBert
    WHEN: I call get_model_structure()
    THEN: the function returns two integers

    """
    assert isinstance(model_structure[0], int)
    assert isinstance(model_structure[1], int)


@pytest.mark.get_model_structure
def test_number_of_heads_and_layers(model_structure):
    """
    Test that get_model_structure() returns the right number of heads and
    layers of ProtBert -- i.e., 16 and 30, respectively.

    GIVEN: attention from ProtBert
    WHEN: I call get_model_structure()
    THEN: the function returns the right number of heads (16) and layers (30)

    """
    assert model_structure[0] == 16
    assert model_structure[1] == 30
