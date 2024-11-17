"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-07

Test suite for the pipeline in preprocess.py.

"""
import pytest
import torch

from ProtACon.modules.basics import all_amino_acids


pytestmark = pytest.mark.preprocess


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
