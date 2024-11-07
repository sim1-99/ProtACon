"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-07

Test suite for preprocess.py.

"""
import numpy as np
import pytest

from ProtACon.modules.basics import (
    CA_Atom,
    all_amino_acids,
    extract_CA_Atoms,
    get_model_structure,
    get_sequence_to_tokenize
)
from ProtACon.preprocess import main


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


def test_CA_Atoms_returns(structure):
    """
    Test that the function extract_CA_Atoms returns a tuple.

    GIVEN: a Bio.PDB.Structure object
    WHEN: I call the function extract_CA_Atoms
    THEN: the function returns a tuple of CA_Atom objects

    """
    CA_Atoms = extract_CA_Atoms(structure)

    assert isinstance(CA_Atoms, tuple)
    assert all(isinstance(atom, CA_Atom) for atom in CA_Atoms)


def test_CA_Atoms_data(structure):
    """
    Test that the CA_Atom objects have correct attributes.

    GIVEN: a Bio.PDB.Structure object
    WHEN: I call the function extract_CA_Atoms
    THEN: the CA_Atom objects have correct attributes

    """
    CA_Atoms = extract_CA_Atoms(structure)

    ''' every amino acid is in the list of the twenty canonical amino acids;
    this also tests that no ligands are present in the CA_Atom objects
    '''
    assert all(atom.name in all_amino_acids for atom in CA_Atoms)
    # the index of the amino acid is a non-negative integer
    assert all(atom.idx >= 0 for atom in CA_Atoms)
    # the coordinates of the amino acid are three floats
    assert all(len(atom.coords) == 3 for atom in CA_Atoms)
    assert all(
        isinstance(coord, np.float32)
        for atom in CA_Atoms for coord in atom.coords
    )
