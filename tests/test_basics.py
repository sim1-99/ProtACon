"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-08

Test suite for dictionaries, lists and classes in basics.py.

"""
import pytest

from ProtACon.modules.basics import (
    CA_Atom,
)

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
