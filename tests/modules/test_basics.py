"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-08

Test suite for dictionaries, lists, classes and functions, in basics.py.

"""
from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra.numpy import arrays
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
    load_Bert,
    normalize_array,
)


pytestmark = pytest.mark.basics
st_array = arrays(
    dtype=float,
    shape=(4, 4),
    elements=st.floats(allow_nan=True, allow_infinity=False, width=16),
).filter(lambda x: len(np.unique(x)) > 2)
# at least 3 unique values to avoid constant arrays (take into account NaNs)


# Dictionaries and lists
@pytest.mark.all_amino_acids
class TestAllAminoAcids:
    """Tests for the list all_amino_acids."""

    def test_all_amino_acids_has_length_twenty(self):
        """
        Test that all_amino_acids -- i.e., the list of the twenty canonical
        amino acids -- has length twenty.

        GIVEN: the list of the twenty canonical amino acids
        THEN: the list has length twenty

        """
        assert len(all_amino_acids) == 20

    def test_all_amino_acids_has_no_duplicates(self):
        """
        Test that all_amino_acids -- i.e., the list of the twenty canonical
        amino acids -- has no duplicates.

        GIVEN: the list of the twenty canonical amino acids
        THEN: the list has no duplicates

        """
        assert len(all_amino_acids) == len(set(all_amino_acids))

    def test_all_amino_acids_are_uppercase(self):
        """
        Test that all_amino_acids -- i.e., the list of the twenty canonical
        amino acids -- is composed of uppercase characters.

        GIVEN: the list of the twenty canonical amino acids
        THEN: the list is composed of uppercase characters

        """
        assert all(char.isupper() for char in all_amino_acids)


@pytest.mark.aa_dicts
class TestAminoAcidDictionaries:
    """Tests for the dictionaries dict_1_to_3 and dict_3_to_1."""

    def test_dict_1_to_3_has_length_twenty(self):
        """
        Test that dict_1_to_3 -- i.e., the dictionary for translating from
        single letter to multiple letter amino acid codes -- has length twenty.

        GIVEN: the dictionary for translating from single letter to multiple
            letter amino acid codes
        THEN: the dictionary has length twenty

        """
        assert len(dict_1_to_3) == 20

    def test_dict_3_to_1_has_length_twenty(self):
        """
        Test that dict_3_to_1 -- i.e., the dictionary for translating from
        multiple letter to single letter amino acid codes -- has length twenty.

        GIVEN: the dictionary for translating from multiple letter to single
            letter amino acid codes
        THEN: the dictionary has length twenty

        """
        assert len(dict_3_to_1) == 20

    def test_dict_1_to_3_and_dict_3_to_1_are_reciprocal(self):
        """
        Test that dict_1_to_3 and dict_3_to_1 are reciprocal.

        GIVEN: the dictionaries for translating from single letter to multiple
            letter amino acid codes and vice versa
        THEN: the dictionaries are reciprocal

        """
        for key, value in dict_1_to_3.items():
            assert dict_3_to_1[value[0]] == key


@pytest.mark.all_amino_acids
@pytest.mark.aa_dicts
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
def test_get_model_structure_returns_ints(tuple_of_3d_4d_tensors):
    """
    Test that get_model_structure() returns two integers.

    GIVEN: a tuple of 3d or 4d torch.Tensor
    WHEN: I call get_model_structure()
    THEN: the function returns two integers

    """
    model_structure = get_model_structure(tuple_of_3d_4d_tensors)

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
    Test that the alphabetic characters in the sequence returned by
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
    Test that the sequence from get_sequence_to_tokenize() is the double as
    long as the tuple of CA_Atom objects, minus one. This is beacuse the chars
    in the sequence are separated by spaces.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the string returned is the double as long as the tuple of CA_Atom
        objects, minus one

    """
    sequence = get_sequence_to_tokenize(tuple_of_CA_Atom)

    # consider the spaces between chars
    assert len(sequence) == len(tuple_of_CA_Atom)*2-1


@pytest.mark.load_Bert
@pytest.mark.usefixtures("mocked_Bert")
def test_load_Bert_returns(model_name):
    """
    Test that load_Bert() returns a tuple storing a model and a tokenizer.

    GIVEN: a string being the name of a model
    WHEN: I call load_Bert()
    THEN: the function returns a tuple storing two objects of type
        transformers.BertModel and transformers.BertTokenizer

    """
    Bert = load_Bert(model_name)

    assert isinstance(Bert, tuple)
    assert isinstance(Bert[0], BertModel)
    assert isinstance(Bert[1], BertTokenizer)


@pytest.mark.normalize_array
def test_normalize_array_returns_array(array_2d):
    """
    Test that normalize_array() returns a numpy array.

    GIVEN: an np.ndarray
    WHEN: I call normalize_array()
    THEN: the function returns an np.ndarray

    """
    assert isinstance(normalize_array(array_2d), np.ndarray)


@pytest.mark.normalize_array
def test_norm_array_shape(array_2d):
    """
    Test that the array returned by normalize_array() has the same shape as the
    input array.

    GIVEN: an np.ndarray
    WHEN: I call normalize_array()
    THEN: the np.ndarray returned has the same shape as the input array

    """
    assert normalize_array(array_2d).shape == array_2d.shape


@pytest.mark.normalize_array
def test_normalize_array_raises_value_error_if_array_is_empty():
    """
    Test that normalize_array() raises a ValueError if the input array is
    empty.

    GIVEN: an empty np.ndarray
    WHEN: I call normalize_array()
    THEN: a ValueError with message "Input array is empty" is raised

    """
    with pytest.raises(ValueError) as excinfo:
        normalize_array(np.array([]))
    assert str(excinfo.value) == "Input array is empty"


@pytest.mark.normalize_array
def test_normalize_array_raises_value_error_if_array_has_all_nans():
    """
    Test that normalize_array() raises a ValueError if the input array has
    all NaN values.

    GIVEN: an np.ndarray with all NaN values
    WHEN: I call normalize_array()
    THEN: a ValueError with message "Input array has all NaN values" is raised

    """
    with pytest.raises(ValueError) as excinfo:
        normalize_array(np.full((4, 4), np.nan))
    assert str(excinfo.value) == "Input array has all NaN values"


@pytest.mark.normalize_array
def test_normalize_array_raises_value_error_if_array_has_const_values():
    """
    Test that normalize_array() raises a ValueError if the input array has
    constant values.

    GIVEN: an np.ndarray with constant values
    WHEN: I call normalize_array()
    THEN: a ValueError with message "Input array has constant values" is raised

    """
    with pytest.raises(ValueError) as excinfo:
        normalize_array(np.full((4, 4), 1))
    assert str(excinfo.value) == "Input array has constant values"


@pytest.mark.normalize_array
def test_nan_values_in_norm_array_are_left_unchanged():
    """
    Test that NaN values in the input array are left unchanged in the array
    returned by normalize_array().

    GIVEN: an np.ndarray
    WHEN: I call normalize_array()
    THEN: NaN values in the input array are left unchanged in the np.ndarray
        returned

    """
    input = np.array(
        [[np.nan, 1, 2],
         [0, np.nan, np.nan],
         [3, np.nan, 5]]
    )
    output = normalize_array(input)

    assert np.all(np.isnan(output), where=np.isnan(input))


@pytest.mark.normalize_array
@given(array=st_array)
def test_norm_array_values_range_from_zero_to_one(array):
    """
    Test that the values in the array returned by normalize_array() are between
    0 and 1, excluding NaN values.

    GIVEN: an np.ndarray
    WHEN: I call normalize_array()
    THEN: the values in the np.ndarray returned are between 0 and 1, excluding
        NaN values

    """
    norm_array = normalize_array(array)

    assert np.all(norm_array >= 0, where=~np.isnan(norm_array))
    assert np.all(norm_array <= 1, where=~np.isnan(norm_array))
