"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-28

Test suite for the functions in contact.py.

"""
from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra.numpy import arrays
import numpy as np
import pytest

from ProtACon.modules.contact import (
    distance_between_atoms,
)


pytestmark = pytest.mark.contact
st_arrays = arrays(
    dtype=float,
    shape=(2, 3),
    elements=st.floats(allow_nan=False, allow_infinity=False, width=16),
)


@pytest.mark.distance_between_atoms
def test_distance_between_atoms_returns_float(tuple_of_CA_Atom):
    """
    Test that distance_between_atoms() returns a float.

    GIVEN: two arrays representing the coordinates of two CA atoms
    WHEN: I call distance_between_atoms()
    THEN: the function returns a float

    """
    output = distance_between_atoms(
        tuple_of_CA_Atom[0].coords, tuple_of_CA_Atom[1].coords
    )
    assert isinstance(output, float)


@pytest.mark.distance_between_atoms
def test_distance_between_atoms_is_zero_for_same_atom(tuple_of_CA_Atom):
    """
    Test that distance_between_atoms() returns zero when called with the same
    atom, as the distance between an atom and itself is zero.

    GIVEN: two arrays representing the coordinates of the same CA atom
    WHEN: I call distance_between_atoms()
    THEN: the float returned is zero

    """
    output = distance_between_atoms(
        tuple_of_CA_Atom[0].coords, tuple_of_CA_Atom[0].coords
    )
    assert output == pytest.approx(0.)


@pytest.mark.distance_between_atoms
@given(arrays=st_arrays)
def test_distance_between_atoms_is_nonnegative(arrays):
    """
    Test that distance_between_atoms() returns a non-negative value, as it
    represents a distance.

    GIVEN: two arrays representing the coordinates of two CA atoms
    WHEN: I call distance_between_atoms()
    THEN: the float returned is greater than or equal to zero

    """
    output = distance_between_atoms(arrays[0], arrays[1])
    assert output >= 0


@pytest.mark.distance_between_atoms
@given(arrays=st_arrays)
def test_distance_AB_is_the_same_as_BA(arrays):
    """
    Test that distance_between_atoms() returns the same value for two atoms
    regardless of their order, as the distance between atom A and atom B is the
    same as the distance between atom B and atom A.

    GIVEN: two arrays representing the coordinates of two CA atoms
    WHEN: I call distance_between_atoms() twice, swapping the order of the
        arrays
    THEN: the float returned is the same

    """
    output_AB = distance_between_atoms(arrays[0], arrays[1])
    output_BA = distance_between_atoms(arrays[1], arrays[0])
    assert output_AB == pytest.approx(output_BA)


