"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-07

Test suite for preprocess.py.

"""
from pathlib import Path
import warnings

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pytest

from ProtACon.modules.basics import (
    CA_Atom,
    all_amino_acids,
    download_pdb,
    extract_CA_atoms,
    get_model_structure,
    get_sequence_to_tokenize
)
from ProtACon.preprocess import main


# Fixtures
@pytest.fixture(scope="module")
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
    Path.unlink(data_path/f"pdb{chain_ID.lower()}.ent")


@pytest.fixture(scope="module")
def CA_atoms(structure):
    """Tuple of the CA_Atom objects of a peptide chain."""
    return extract_CA_atoms(structure)


@pytest.fixture(scope="module")
def sequence(CA_atoms):
    """String with the amino acids of the residues in a peptide chain."""
    return get_sequence_to_tokenize(CA_atoms)


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


def test_sequence_is_string(sequence):
    """
    Test that get_sequence_to_tokenize() returns a string.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the function returns a string

    """
    assert isinstance(sequence, str)


def test_chars_are_alpha_and_spaces(sequence):
    """
    Test that the sequence from get_sequence_to_tokenize() is composed of
    alphabetic characters and spaces.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the string returned is composed of alphabetic characters and spaces

    """
    assert all(char.isalpha() or char == " " for char in sequence)


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


def test_sequence_length(CA_atoms, sequence):
    """
    Test that the sequence from get_sequence_to_tokenize() has the right
    length.

    GIVEN: a tuple of CA_Atom objects
    WHEN: I call get_sequence_to_tokenize()
    THEN: the string returned has the right length

    """
    assert len(sequence) == len(CA_atoms)*2-1  # consider spaces between chars
