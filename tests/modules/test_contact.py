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
    binarize_contact_map,
    distance_between_atoms,
    generate_distance_map,
)


pytestmark = pytest.mark.contact
param_dist_cutoff = [0.0, 1.5, 4.3, 8.0, 10.0]
param_pos_cutoff = [1, 2, 3, 4]
st_arrays = arrays(
    dtype=float,
    shape=(2, 3),
    elements=st.floats(allow_nan=False, allow_infinity=False, width=16),
)


@pytest.mark.binarize_contact_map
def test_binarize_contact_map_returns_array(dist_map):
    """
    Test that binarize_contact_map() returns a numpy array.

    GIVEN: an np.ndarray, a float and an int
    WHEN: I call binarize_contact_map()
    THEN: the function returns an np.ndarray

    """
    output = binarize_contact_map(dist_map, 8.0, 6)
    assert isinstance(output, np.ndarray)


@pytest.mark.binarize_contact_map
def test_contact_map_shape(dist_map):
    """
    Test that the array returned by binarize_contact_map() has the same shape
    as the input distance map.

    GIVEN: an np.ndarray, a float and an int
    WHEN: I call binarize_contact_map()
    THEN: the np.ndarray returned has the same shape as the input np.ndarray

    """
    output = binarize_contact_map(dist_map, 8.0, 6)
    assert output.shape == dist_map.shape


@pytest.mark.binarize_contact_map
def test_contact_map_values_are_zero_or_one(dist_map):
    """
    Test that the array returned by binarize_contact_map() contains only zeros
    and ones, as it represents a binary contact map.

    GIVEN: an np.ndarray, a float and an int
    WHEN: I call binarize_contact_map()
    THEN: all the values in the np.ndarray returned are either zero or one

    """
    output = binarize_contact_map(dist_map, 8.0, 6)
    assert np.all(np.isin(output, [0, 1]))


@pytest.mark.binarize_contact_map
def test_contact_map_is_symmetric(dist_map):
    """
    Test that the array returned by binarize_contact_map() is symmetric.

    GIVEN: an np.ndarray, a float and an int
    WHEN: I call binarize_contact_map()
    THEN: the np.ndarray returned is symmetric

    """
    output = binarize_contact_map(dist_map, 8.0, 6)
    assert np.all(output == output.T)


@pytest.mark.binarize_contact_map
@pytest.mark.parametrize("pos_cutoff", param_pos_cutoff)
def test_contact_map_values_around_the_diagonal_are_zero(
    dist_map, pos_cutoff,
):
    """
    Test that the values in the array returned by binarize_contact_map() are
    zero on the diagnonal and around it, in a neighborhood of size equal to the
    position cutoff.

    Given that the fixture dist_map is a 4x4 matrix, and that the values are
    set to zero if the absolute difference between the indices is strictly less
    than the position cutoff, then pos_cutoff can range from 1 to 4. The cutoff
    on the distance is set to 0.0 to avoid influencing the test.

    GIVEN: an np.ndarray, a float and an int
    WHEN: I call binarize_contact_map()
    THEN: the values in a neighborhood of the diagonal equal to pos_cutoff are
        zero

    """
    output = binarize_contact_map(dist_map, 0, pos_cutoff)
    diag = [np.diag(output, k) for k in range(-pos_cutoff+1, pos_cutoff-1)]
    assert all(np.all(diag_item == 0) for diag_item in diag)


@pytest.mark.binarize_contact_map
@pytest.mark.parametrize("dist_cutoff", param_dist_cutoff)
def test_contact_map_values_above_dist_cutoff_are_zero(dist_cutoff, dist_map):
    """
    Test that the values in the distance map that are above the cutoff are set
    to zero in the contact map returned by binarize_contact_map().

    The cutoff on the position is set to zero to avoid influencing the test.

    GIVEN: an np.ndarray, a float and an int
    WHEN: I call binarize_contact_map()
    THEN: the values higher than dist_cutoff are set to zero in the np.ndarray
        returned

    """
    output = binarize_contact_map(dist_map, dist_cutoff, 0)
    high_values = dist_map > dist_cutoff
    assert np.all(output[high_values] == 0)


@pytest.mark.binarize_contact_map
@pytest.mark.parametrize("dist_cutoff", param_dist_cutoff)
def test_contact_map_values_below_dist_cutoff_are_one(dist_cutoff, dist_map):
    """
    Test that the values in the distance map that are below the cutoff are set
    to one in the contact map returned by binarize_contact_map().

    The cutoff on the position is set to zero to avoid influencing the test.

    GIVEN: an np.ndarray, a float and an int
    WHEN: I call binarize_contact_map()
    THEN: the values lower than dist_cutoff are set to one in the np.ndarray
        returned

    """
    output = binarize_contact_map(dist_map, dist_cutoff, 0)
    low_values = (dist_map <= dist_cutoff) & (dist_map > 0)
    assert np.all(output[low_values] == 1)


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


@pytest.mark.generate_distance_map
def test_generate_distance_map_returns_array(tuple_of_CA_Atom):
    """
    Test that generate_distance_map() returns a numpy array.

    GIVEN: a tuple of CA_Atom
    WHEN: I call generate_distance_map()
    THEN: the function returns an np.ndarray

    """
    output = generate_distance_map(tuple_of_CA_Atom)
    assert isinstance(output, np.ndarray)


@pytest.mark.generate_distance_map
def test_distance_map_shape(tuple_of_CA_Atom):
    """
    Test that the array returned by generate_distance_map() is a square matrix
    with sides equal to the number of elements in tuple_of_CA_Atom.

    GIVEN: a tuple of CA_Atom
    WHEN: I call generate_distance_map()
    THEN: the np.ndarray returned has shape (len(tuple_of_CA_Atom),
        len(tuple_of_CA_Atom))

    """
    output = generate_distance_map(tuple_of_CA_Atom)
    assert output.shape == (len(tuple_of_CA_Atom), len(tuple_of_CA_Atom))


@pytest.mark.generate_distance_map
def test_distance_map_values_are_nonnegative(tuple_of_CA_Atom):
    """
    Test that the array returned by generate_distance_map() contains only
    non-negative values, as it represents euclidean distances.

    GIVEN: a tuple of CA_Atom
    WHEN: I call generate_distance_map()
    THEN: all the values in the np.ndarray returned are greater than or equal
        to zero

    """
    output = generate_distance_map(tuple_of_CA_Atom)
    assert np.all(output >= 0)


@pytest.mark.generate_distance_map
def test_distance_map_diagonal_is_zero(tuple_of_CA_Atom):
    """
    Test that the diagonal of the array returned by generate_distance_map() is
    zero, as the distance between a residue and itself is zero.

    GIVEN: a tuple of CA_Atom
    WHEN: I call generate_distance_map()
    THEN: the diagonal of the np.ndarray returned is zero

    """
    output = generate_distance_map(tuple_of_CA_Atom)
    assert np.all(np.diag(output) == 0)


@pytest.mark.generate_distance_map
def test_distance_map_is_symmetric(tuple_of_CA_Atom):
    """
    Test that the array returned by generate_distance_map() is symmetric, as
    the distance between residue A and residue B is the same as the distance
    between residue B and residue A.

    GIVEN: a tuple of CA_Atom
    WHEN: I call generate_distance_map()
    THEN: the np.ndarray returned is symmetric

    """
    output = generate_distance_map(tuple_of_CA_Atom)
    assert np.all(output == output.T)
