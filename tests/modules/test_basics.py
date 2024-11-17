"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-08

Test suite for dictionaries, lists, classes and functions, in basics.py.

"""
from transformers import BertModel, BertTokenizer
import numpy as np
import pytest

from ProtACon.modules.basics import (
    CA_Atom,
    all_amino_acids,
    dict_1_to_3,
    dict_3_to_1,
    extract_CA_atoms,
    get_model_structure,
    get_sequence_to_tokenize,
    load_model,
)


pytestmark = pytest.mark.basics


# Dictionaries and lists
@pytest.mark.all_amino_acids
def test_all_amino_acids_has_length_twenty():
    """
    Test that all_amino_acids -- i.e. the list of the twenty canonical amino
    acids -- has length twenty.

    GIVEN: the list of the twenty canonical amino acids
    THEN: the list has length twenty

    """
    assert len(all_amino_acids) == 20


@pytest.mark.all_amino_acids
def test_all_amino_acids_has_no_duplicates():
    """
    Test that all_amino_acids -- i.e. the list of the twenty canonical amino
    acids -- has no duplicates.

    GIVEN: the list of the twenty canonical amino acids
    THEN: the list has no duplicates

    """
    assert len(all_amino_acids) == len(set(all_amino_acids))


@pytest.mark.all_amino_acids
def test_all_amino_acids_are_uppercase():
    """
    Test that all_amino_acids -- i.e. the list of the twenty canonical amino
    acids -- is composed of uppercase characters.

    GIVEN: the list of the twenty canonical amino acids
    THEN: the list is composed of uppercase characters

    """
    assert all(char.isupper() for char in all_amino_acids)


@pytest.mark.dict_1_to_3
def test_dict_1_to_3_has_length_twenty():
    """
    Test that dict_1_to_3 -- i.e. the dictionary for translating from single
    letter to multiple letter amino acid codes -- has length twenty.

    GIVEN: the dictionary for translating from single letter to multiple letter
        amino acid codes
    THEN: the dictionary has length twenty

    """
    assert len(dict_1_to_3) == 20


@pytest.mark.dict_3_to_1
def test_dict_3_to_1_has_length_twenty():
    """
    Test that dict_3_to_1 -- i.e. the dictionary for translating from multiple
    letter to single letter amino acid codes -- has length twenty.

    GIVEN: the dictionary for translating from multiple letter to single letter
        amino acid codes
    THEN: the dictionary has length twenty

    """
    assert len(dict_3_to_1) == 20


@pytest.mark.dict_1_to_3
@pytest.mark.dict_3_to_1
def test_dict_1_to_3_and_dict_3_to_1_are_reciprocal():
    """
    Test that dict_1_to_3 and dict_3_to_1 are reciprocal.

    GIVEN: the dictionaries for translating from single letter to multiple
        letter amino acid codes and vice versa
    THEN: the dictionaries are reciprocal

    """
    for key, value in dict_1_to_3.items():
        assert dict_3_to_1[value[0]] == key


@pytest.mark.all_amino_acids
@pytest.mark.dict_1_to_3
@pytest.mark.dict_3_to_1
def test_all_amino_acids_in_dictionaries():
    """
    Test that the items in all_amino_acids are in the dictionaries for
    translating from single letter to multiple letter amino acid codes and vice
    versa.

    GIVEN: the list of the twenty canonical amino acids and the dictionaries
        for translating from single letter to multiple letter amino acid codes
        and vice versa
    THEN: all the amino acids are in the dictionaries

    """
    for amino_acid in all_amino_acids:
        assert amino_acid in dict_1_to_3.keys()
        assert amino_acid in dict_3_to_1.values()


# Classes
@pytest.mark.CA_Atom
def test_CA_Atom_init():
    """
    Test the creation of an instance of the CA_Atom class.

    GIVEN: amino acid, position and coordinates of a residue in a peptide chain
    WHEN: a CA_Atom object is instantiated
    THEN: the object has the correct attributes

    """
    atom = CA_Atom(name="M", idx=5, coords=[0.0, -2.0, 11.0])

    assert atom.name == "M"
    assert atom.idx == 5
    assert atom.coords == [0.0, -2.0, 11.0]


# Functions
@pytest.mark.extract_CA_atoms
def test_extract_CA_atoms_returns_tuple_of_CA_Atom(structure):
    """
    Test that extract_CA_atoms() returns a tuple of CA_Atom objects.

    GIVEN: a Bio.PDB.Structure object
    WHEN: I call extract_CA_atoms()
    THEN: the function returns a tuple of CA_Atom objects

    """
    CA_atoms = extract_CA_atoms(structure)

    assert isinstance(CA_atoms, tuple)
    assert all(isinstance(atom, CA_Atom) for atom in CA_atoms)


@pytest.mark.extract_CA_atoms
def test_CA_atoms_data(structure):
    """
    Test that the CA_Atom objects in the tuple from extract_CA_atoms() have
    correct attributes.

    GIVEN: a Bio.PDB.Structure object
    WHEN: I call extract_CA_atoms()
    THEN: the CA_Atom objects in the tuple returned have correct attributes

    """
    CA_atoms = extract_CA_atoms(structure)

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


@pytest.mark.get_model_structure
def test_get_model_structure_returns_ints(tuple_of_tensors):
    """
    Test that get_model_structure() returns two integers.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call get_model_structure()
    THEN: the function returns two integers

    """
    model_structure = get_model_structure(tuple_of_tensors)

    assert isinstance(model_structure[0], int)
    assert isinstance(model_structure[1], int)


@pytest.mark.get_sequence_to_tokenize
def test_get_sequence_to_tokenize_returns_string(tuple_of_CA_Atom):
    """
    Test that get_sequence_to_tokenize() returns a string.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the function returns a string

    """
    sequence = get_sequence_to_tokenize(tuple_of_CA_Atom)

    assert isinstance(sequence, str)


@pytest.mark.get_sequence_to_tokenize
def test_spaces_between_chars(tuple_of_CA_Atom):
    """
    Test that the alphabetic characters in the sequence from
    get_sequence_to_tokenize() are separated with spaces.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the alphabetic characters in the string returned are separated with
        spaces

    """
    sequence = get_sequence_to_tokenize(tuple_of_CA_Atom)

    assert all(char.isalpha() for i, char in enumerate(sequence) if i % 2 == 0)


@pytest.mark.get_sequence_to_tokenize
def test_sequence_length(tuple_of_CA_Atom):
    """
    Test that the sequence from get_sequence_to_tokenize() has the right
    length.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the string returned has the right length

    """
    sequence = get_sequence_to_tokenize(tuple_of_CA_Atom)

    # consider the spaces between chars
    assert len(sequence) == len(tuple_of_CA_Atom)*2-1


@pytest.mark.load_model
def test_load_model_returns(model_name):
    """
    Test that load_model() returns a tuple storing a model and a tokenizer.

    GIVEN: a string being the name of a model
    WHEN: I call load_model()
    THEN: the function returns a tuple storing two objects of type
        transformers.BertModel and transformers.BertTokenizer

    """
    model = load_model(model_name)

    assert isinstance(model, tuple)
    assert isinstance(model[0], BertModel)
    assert isinstance(model[1], BertTokenizer)
