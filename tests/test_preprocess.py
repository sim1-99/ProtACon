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
