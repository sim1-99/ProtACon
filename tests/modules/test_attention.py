"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-13

Test suite for the functions in attention.py.

"""
from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra.numpy import arrays
import numpy as np
import pandas as pd
import pytest
import torch

from ProtACon.modules.attention import (
    average_matrices_together,
    clean_attention,
    compute_attention_alignment,
    compute_attention_similarity,
    get_amino_acid_pos,
    get_attention_to_amino_acid,
    include_att_to_missing_aa,
    sum_attention_on_columns,
    sum_attention_on_heads,
    threshold_attention,
)
from ProtACon.modules.basics import all_amino_acids


pytestmark = pytest.mark.attention

# Test parameters
param_aa = ["A", "M", "L", "V", "Y", "D"]
param_cutoff = [0.0, 0.5, 0.9, 1.1]
param_df_idx = list(range(3))

# Hypothesis strategies
"""min_value and max_value respectively represent the minimum and maximum
numbers of residues in the chain -- i.e., the possible length of the sides of
the square attention matrix. 1 is the minimum reasonable number of heads, while
5 is set to limit the execution time.
"""
st_matrix_dim = st.lists(
    elements=st.integers(min_value=1, max_value=5),
    min_size=2,  # in case of attention averages
    max_size=3,  # in case of attention matrices
).filter(lambda x: x[-2] == x[-1])  # attention is always a square matrix


@st.composite
def draw_bin_array(draw):
    """
    Return a binary 2d np.ndarray generated with a hypothesis strategy.

    The array returned can be used within a @given decorator before a test
    function.
    The dimensions of the array generated with the strategy vary together with
    the last two dimensions of the tensors returned by draw_bin_array().

    """
    # the array and tensor dimensions must match to compute attention alignment
    matrix_dim = draw(st.shared(st_matrix_dim, key="tensor_dim"))
    st_bin_array = arrays(
        dtype=int,
        shape=matrix_dim[-2:],  # same shape as the attention matrix
        elements=st.integers(min_value=0, max_value=1),  # the map is binary
    )

    return draw(st_bin_array)


@st.composite
def draw_tensors(draw):
    """
    Return a tuple of two torch.Tensor generated with a hypothesis strategy.

    The tuple returned can be used within a @given decorator before a test
    function.
    The last two dimensions of the tensors generated with the strategy vary
    together with the dimensions of the array returned by draw_bin_array().

    """
    # the array and tensor dimensions must match to compute attention alignment
    matrix_dim = draw(st.shared(st_matrix_dim, key="tensor_dim"))
    st_arrays = arrays(
        dtype=float,
        shape=matrix_dim,
        elements=st.floats(
            min_value=0,
            max_value=1,  # chosen to limit execution time
            allow_nan=False,
            allow_infinity=False,
            width=16,
        ),
    )

    return tuple(torch.from_numpy(draw(st_arrays)) for _ in range(2))


# Tests
@pytest.mark.average_matrices_together
def test_average_matrices_together_returns_tuple_of_tensors(tuple_of_tensors):
    """
    Test that average_matrices_together() returns a tuple of torch.Tensor.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call average_matrices_together()
    THEN: the function returns a tuple of torch.Tensor

    """
    output = average_matrices_together(tuple_of_tensors)

    assert isinstance(output, tuple)
    assert all(isinstance(tensor, torch.Tensor) for tensor in output)


@pytest.mark.average_matrices_together
def test_att_avgs_len(tuple_of_tensors):
    """
    Test that the tuple returned by average_matrices_together() has the same
    length as the input tuple plus one, because the average of the averages is
    added as last element.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call average_matrices_together()
    THEN: the tuple returned has the same length as the input tuple plus one

    """
    output = average_matrices_together(tuple_of_tensors)

    assert len(output) == len(tuple_of_tensors)+1


@pytest.mark.average_matrices_together
def test_att_avgs_shape(tuple_of_tensors):
    """
    Test that the tensors returned by average_matrices_together() have shape
    equal to the last two dimensions of the input tensors.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call average_matrices_together()
    THEN: the tensors returned have shape equal to the last two dimensions of
        the input tensors

    """
    output = average_matrices_together(tuple_of_tensors)

    for t_in, t_out in zip(tuple_of_tensors, output):
        assert t_out.shape == t_in.shape[-2:]
        assert output[-1].shape == t_in.shape[-2:]


@pytest.mark.average_matrices_together
def test_att_avgs_are_averages(tuple_of_tensors):
    """
    Test that the values in the tensors returned by average_matrices_together()
    are equal to the average of the values in the corresponding positions of
    the input tensors.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call average_matrices_together()
    THEN: the values in the tensors returned are equal to the average of the
        values in the corresponding positions of the input tensors

    """
    output = average_matrices_together(tuple_of_tensors)

    for t_in, t_out in zip(tuple_of_tensors, output):
        # take into account possible batch dimension in input tensors
        t_in = torch.flatten(t_in, end_dim=-3)
        assert all(
            t_out[row_idx, col_idx] == pytest.approx(
                sum(head[row_idx, col_idx] for head in t_in)/t_in.shape[0]
            )
            for row_idx in range(t_in.shape[-2])
            for col_idx in range(t_in.shape[-1])
        )


@pytest.mark.average_matrices_together
def test_last_tensor_is_average_of_averages(tuple_of_tensors):
    """
    Test that the values in the last tensor returned by
    average_matrices_together() are equal to the average of the values in the
    other output tensors.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call average_matrices_together()
    THEN: the values in the last tensor returned are equal to the average of
        the values in the other output tensors

    """
    output = average_matrices_together(tuple_of_tensors)
    last_tensor = output[-1]

    assert all(
        last_tensor[row_idx, col_idx] == pytest.approx(
            sum(layer[row_idx, col_idx] for layer in output[:-1]) /
            len(output[:-1])
        )
        for row_idx in range(last_tensor.shape[0])
        for col_idx in range(last_tensor.shape[1])
    )


@pytest.mark.clean_attention
def test_clean_attention_returns_tuple_of_tensors(tuple_of_tensors):
    """
    Test that clean_attention() returns a tuple of torch.Tensor.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call clean_attention()
    THEN: the function returns a tuple of torch.Tensor

    """
    output = clean_attention(tuple_of_tensors)

    assert isinstance(output, tuple)
    assert all(isinstance(tensor, torch.Tensor) for tensor in output)


@pytest.mark.clean_attention
def test_clean_attention_raises_value_error_on_unexpected_shape():
    """
    Test that clean_attention() raises a ValueError if the input tensors have
    a number of dimensions different from 3 or 4.

    GIVEN: a tuple of torch.Tensor with a number of dimensions different from 3
        or 4
    WHEN: I call clean_attention()
    THEN: a ValueError is raised

    """
    with pytest.raises(ValueError):
        clean_attention(
            (torch.rand(2, 3), torch.rand(1, 3, 4, 4, 5), torch.rand(1))
        )


@pytest.mark.clean_attention
def test_cleaned_attention_len(tuple_of_tensors):
    """
    Test that the tuple returned by clean_attention() has the same length as
    the input tuple.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call clean_attention()
    THEN: the tuple returned has the same length as the input tuple

    """
    output = clean_attention(tuple_of_tensors)

    assert len(output) == len(tuple_of_tensors)


@pytest.mark.clean_attention
def test_cleaned_attention_shape(tuple_of_tensors):
    """
    Having the input tensors shape (batch_size, n_heads, seq_len+2, seq_len+2)
    and/or (n_heads, seq_len+2, seq_len+2), test that the tensors returned by
    clean_attention() have shape (n_heads, seq_len, seq_len).

    GIVEN: a tuple of torch.Tensor with shape (batch_size, n_heads, seq_len+2,
        seq_len+2) and/or (n_heads, seq_len+2, seq_len+2)
    WHEN: I call clean_attention()
    THEN: the tensors in the tuple returned have shape (n_heads, seq_len,
        seq_len)

    """
    output = clean_attention(tuple_of_tensors)

    for t_in, t_out in zip(tuple_of_tensors, output):
        # flattening not beyond the third to last dimension
        t_in = torch.flatten(t_in, end_dim=-3)
        assert t_out.shape[-3] == t_in.shape[-3]
        assert t_out.shape[-2] == t_in.shape[-2]-2
        assert t_out.shape[-1] == t_in.shape[-1]-2


@pytest.mark.clean_attention
def test_cleaned_attention_sums(tuple_of_tensors):
    """
    Test that the sum of the values in each tensor returned by
    clean_attention() is equal to the sum in the corresponding input tensor
    minus the attention values from the first and the last tokens.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call clean_attention()
    THEN: the sum of the values in each output tensor is equal to the sum of
        the values in the corresponding input tensor minus the values from the
        first and the last rows and the first and the last columns of the
        tensor

    """
    output = clean_attention(tuple_of_tensors)

    for t_in, t_out in zip(tuple_of_tensors, output):
        # flattening not beyond the third to last dimension
        t_in = torch.flatten(t_in, end_dim=-3)
        assert torch.sum(t_out) == pytest.approx(
            torch.sum(t_in[:, 1:-1, 1:-1])
        )


@pytest.mark.compute_attention_alignment
def test_compute_attention_alignment_returns_array(
    bin_array_2d, tuple_of_tensors
):
    """
    Test that compute_attention_alignment() returns a numpy array.

    GIVEN: a tuple of torch.Tensor and an np.ndarray
    WHEN: I call compute_attention_alignment()
    THEN: the function returns a np.ndarray

    """
    output = compute_attention_alignment(tuple_of_tensors, bin_array_2d)

    assert isinstance(output, np.ndarray)


@pytest.mark.compute_attention_alignment
def test_att_align_shape_with_att_matrices(
    bin_array_2d, n_heads, n_layers, tuple_of_tensors
):
    """
    Test that the array returned by compute_attention_alignment() has shape
    (n_layers, n_heads), if the input tensors have 3 or 4 dimensions.

    GIVEN: a tuple of torch.Tensor with 3 or 4 dimensions and an np.ndarray
    WHEN: I call compute_attention_alignment()
    THEN: the np.ndarray returned has shape (n_layers, n_heads)

    """
    output = compute_attention_alignment(tuple_of_tensors, bin_array_2d)

    assert output.shape == (n_layers, n_heads)


@pytest.mark.compute_attention_alignment
def test_att_align_shape_with_att_avgs(bin_array_2d, n_layers):
    """
    Test that the array returned by compute_attention_alignment() has shape
    (n_layers), if the input tensors have 2 dimensions.

    GIVEN: a tuple of torch.Tensor with 2 dimensions and an np.ndarray
    WHEN: I call compute_attention_alignment()
    THEN: the np.ndarray returned has shape (n_layers)

    """
    tuple_of_tensors = (
        torch.tensor(  # shape = (4, 4)
            [[0.1, 0.8, 0.3, 0.6], [0.5, 0.9, 0.7, 0.7],
             [0.2, 0.4, 0.1, 0.0], [0.3, 0.5, 0.6, 0.8]]
        ),
        torch.tensor(  # shape = (4, 4)
            [[0.2, 0.7, 0.4, 0.5], [0.6, 0.8, 0.6, 0.8],
             [0.1, 0.3, 0.2, 0.1], [0.4, 0.6, 0.5, 0.7]]
        ),
    )
    output = compute_attention_alignment(tuple_of_tensors, bin_array_2d)

    assert output.shape == (n_layers,)


@pytest.mark.compute_attention_alignment
@given(bin_array_2d=draw_bin_array(), tuple_of_tensors=draw_tensors())
def test_att_align_ranges_from_zero_to_one(bin_array_2d, tuple_of_tensors):
    """
    Test that the values in the array returned by compute_attention_alignment()
    are between zero and one.

    GIVEN: a tuple of torch.Tensor and an np.ndarray
    WHEN: I call compute_attention_alignment()
    THEN: the values in the np.ndarray returned are between zero and one

    """
    output = compute_attention_alignment(tuple_of_tensors, bin_array_2d)

    assert np.all(output >= 0)
    assert np.all(output <= 1)


@pytest.mark.compute_attention_alignment
def test_att_align_is_sum_of_3d_4d_arrays_where_map_is_one(
    bin_array_2d, tuple_of_tensors
):
    """
    Test that the values in the array returned by compute_attention_alignment()
    are equal to the sum of the values in each square sub-matrix of 3d/4d input
    tensors, in the positions where the binary map is one.

    GIVEN: a tuple of 3d/4d torch.Tensor and an np.ndarray
    WHEN: I call compute_attention_alignment()
    THEN: the values in the np.ndarray returned are equal to the sum of the
        values in each square sub-matrix of the input tensors, in the positions
        where the binary map is one

    """
    output = compute_attention_alignment(tuple_of_tensors, bin_array_2d)

    for t_idx, t in enumerate(tuple_of_tensors):
        """flattening not beyond the third to last dimension makes the test
        valid both for tensors with batch dimension and without it
        """
        t = torch.flatten(t, end_dim=-3)
        for dim_idx in range(t.shape[-3]):
            assert output[t_idx, dim_idx] == pytest.approx(
                torch.sum(t[dim_idx]*bin_array_2d)/torch.sum(t[dim_idx])
            )


@pytest.mark.compute_attention_alignment
def test_att_align_is_sum_of_2d_arrays_where_map_is_one(
    bin_array_2d, tuple_of_tensors
):
    """
    Test that the values in the array returned by compute_attention_alignment()
    are equal to the sum of the values in each square sub-matrix of 2d input
    tensors, in the positions where the binary map is one.

    GIVEN: a tuple of 2d torch.Tensor and an np.ndarray
    WHEN: I call compute_attention_alignment()
    THEN: the values in the np.ndarray returned are equal to the sum of the
        values in each square sub-matrix of the input tensors, in the positions
        where the binary map is one

    """
    tuple_of_tensors = (tuple_of_tensors[1][0], tuple_of_tensors[1][1])
    output = compute_attention_alignment(tuple_of_tensors, bin_array_2d)

    for t_idx, t in enumerate(tuple_of_tensors):
        assert output[t_idx] == pytest.approx(
            torch.sum(t*bin_array_2d)/torch.sum(t)
        )


@pytest.mark.compute_attention_similarity
def test_compute_attention_similarity_returns_data_frame(
    amino_acids_in_chain, T_att_to_aa
):
    """
    Test that compute_attention_similarity() returns a pandas DataFrame.

    GIVEN: a torch.Tensor and a list of strings
    WHEN: I call compute_attention_similarity()
    THEN: the function returns a pd.DataFrame

    """
    output = compute_attention_similarity(T_att_to_aa, amino_acids_in_chain)

    assert isinstance(output, pd.DataFrame)


@pytest.mark.compute_attention_similarity
def test_att_sim_shape(amino_acids_in_chain, T_att_to_aa):
    """
    Test that the data frame returned by compute_attention_similarity() has
    shape (len(amino_acids_in_chain), len(amino_acids_in_chain)).

    GIVEN: a torch.Tensor and a list of strings
    WHEN: I call compute_attention_similarity()
    THEN: the pd.DataFrame returned has shape (len(amino_acids_in_chain),
        len(amino_acids_in_chain))

    """
    output = compute_attention_similarity(T_att_to_aa, amino_acids_in_chain)

    assert output.shape == (
        len(amino_acids_in_chain), len(amino_acids_in_chain)
    )


@pytest.mark.compute_attention_similarity
def test_att_sim_is_symmetric(amino_acids_in_chain, T_att_to_aa):
    """
    Test that the data frame returned by compute_attention_similarity() is
    symmetric.

    GIVEN: a torch.Tensor and a list of strings
    WHEN: I call compute_attention_similarity()
    THEN: the pd.DataFrame returned is symmetric

    """
    output = compute_attention_similarity(T_att_to_aa, amino_acids_in_chain)

    assert output.equals(output.T)


@pytest.mark.compute_attention_similarity
def test_att_sim_diagonal_is_missing(amino_acids_in_chain, T_att_to_aa):
    """
    Test that the values in the diagonal of the data frame returned by
    compute_attention_similarity() are missing.

    GIVEN: a torch.Tensor and a list of strings
    WHEN: I call compute_attention_similarity()
    THEN: the values in the diagonal of the pd.DataFrame returned are missing

    """
    output = compute_attention_similarity(T_att_to_aa, amino_acids_in_chain)

    assert all(pd.isnull(output.iloc[idx, idx]) for idx in range(len(output)))


@pytest.mark.compute_attention_similarity
def test_att_sim_ranges_from_zero_to_one(amino_acids_in_chain, T_att_to_aa):
    """
    Test that the values in the data frame returned by
    compute_attention_similarity() are between zero and one, excluding missing
    values.

    GIVEN: a torch.Tensor and a list of strings
    WHEN: I call compute_attention_similarity()
    THEN: the values in the pd.DataFrame returned are between zero and one,
        excluding missing values

    """
    output = compute_attention_similarity(T_att_to_aa, amino_acids_in_chain)

    assert output.all(axis=None, skipna=True) >= 0
    assert output.all(axis=None, skipna=True) <= 1


@pytest.mark.get_amino_acid_pos
def test_get_amino_acid_pos_returns_list_of_integers(tokens):
    """
    Test that get_amino_acid_pos() returns a list of integers.

    GIVEN: an amino acid and a list of tokens
    WHEN: I call get_amino_acid_pos()
    THEN: the function returns a list of integers

    """
    output = get_amino_acid_pos("A", tokens)

    assert isinstance(output, list)
    assert all(isinstance(pos, int) for pos in output)


@pytest.mark.get_amino_acid_pos
@pytest.mark.parametrize("amino_acid", param_aa)
def test_amino_acid_positions_are_in_range(amino_acid, tokens):
    """
    Test that the positions returned by get_amino_acid_pos() for each amino
    acid are within the range of the list of tokens.

    GIVEN: an amino acid and a list of tokens
    WHEN: I call get_amino_acid_pos()
    THEN: the positions returned are within the range of the list of tokens

    """
    output = get_amino_acid_pos(amino_acid, tokens)

    assert all(0 <= pos < len(tokens) for pos in output)


@pytest.mark.get_amino_acid_pos
@pytest.mark.parametrize("amino_acid", param_aa)
def test_amino_acid_positions_are_unique(amino_acid, tokens):
    """
    Test that the positions returned by get_amino_acid_pos() for each amino
    acid are unique -- i.e., no integer is repeated in the list.

    GIVEN: an amino acid and a list of tokens
    WHEN: I call get_amino_acid_pos()
    THEN: no integer is repeated in the list of positions returned

    """
    output = get_amino_acid_pos(amino_acid, tokens)

    assert len(output) == len(set(output))


@pytest.mark.get_amino_acid_pos
@pytest.mark.parametrize("amino_acid", param_aa)
def test_amino_acid_positions_are_correct(amino_acid, tokens):
    """
    Test that the positions returned by get_amino_acid_pos() for each amino
    acid correspond to the same amino acid in the list of tokens.

    GIVEN: an amino acid and a list of tokens
    WHEN: I call get_amino_acid_pos()
    THEN: the positions returned correspond to the same amino acid in the list
        of tokens

    """
    output = get_amino_acid_pos(amino_acid, tokens)

    assert all(tokens[pos] == amino_acid for pos in output)


@pytest.mark.get_amino_acid_pos
@pytest.mark.parametrize("amino_acid", param_aa)
def test_no_position_is_missing_for_amino_acid(amino_acid, tokens):
    """
    Test that the positions of a given amino acid in the list of tokens are
    all returned by get_amino_acid_pos().

    GIVEN: an amino acid and a list of tokens
    WHEN: I call get_amino_acid_pos()
    THEN: the function returns all the positions of the amino acid in the list
        of tokens

    """
    output = get_amino_acid_pos(amino_acid, tokens)

    assert len(output) == tokens.count(amino_acid)


@pytest.mark.get_amino_acid_pos
def test_get_amino_acid_positions_is_empty_for_missing_amino_acid(tokens):
    """
    Test that get_amino_acid_pos() returns an empty list when the amino acid
    is not present in the list of tokens.

    GIVEN: an amino acid not present in the list of tokens and a list of tokens
    WHEN: I call get_amino_acid_pos()
    THEN: the function returns an empty list

    """
    output = get_amino_acid_pos("Z", tokens)

    assert isinstance(output, list)
    assert len(output) == 0


@pytest.mark.sum_attention_on_columns
def test_sum_attention_on_columns_returns_list_of_tensors(tuple_of_tensors):
    """
    Test that sum_attention_on_columns() returns a list of torch.Tensor.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call sum_attention_on_columns()
    THEN: the function returns a list of torch.Tensor

    """
    output = sum_attention_on_columns(tuple_of_tensors)

    assert isinstance(output, list)
    assert all(isinstance(tensor, torch.Tensor) for tensor in output)


@pytest.mark.get_attention_to_amino_acid
def test_get_attention_to_amino_acid_returns_tensor(
    amino_acid_df, attention_column_sums, n_heads, n_layers
):
    """
    Test that get_attention_to_amino_acid() returns a torch.Tensor.

    GIVEN: a list of torch.Tensor, a list of integers, and two integers
    WHEN: I call get_attention_to_amino_acid()
    THEN: the function returns a torch.Tensor

    """
    output = get_attention_to_amino_acid(
        attention_column_sums,
        amino_acid_df.at[0, "Position in Token List"],
        n_heads,
        n_layers,
    )

    assert isinstance(output, torch.Tensor)


@pytest.mark.get_attention_to_amino_acid
def test_attention_to_amino_acid_shape(
    amino_acid_df, attention_column_sums, n_heads, n_layers
):
    """
    Test that the tensor returned by get_attention_to_amino_acid() has shape
    (n_layers, n_heads).

    GIVEN: a list of torch.Tensor, a list of integers, and two integers
    WHEN: I call get_attention_to_amino_acid()
    THEN: the tensor returned has shape (n_layers, n_heads)

    """
    output = get_attention_to_amino_acid(
        attention_column_sums,
        amino_acid_df.at[0, "Position in Token List"],
        n_heads,
        n_layers,
    )

    assert output.shape == (n_layers, n_heads)


@pytest.mark.get_attention_to_amino_acid
@pytest.mark.parametrize("df_idx", param_df_idx)
def test_attention_to_aa_is_sum_of_aa_columns(
    amino_acid_df, attention_column_sums, df_idx, n_heads, n_layers
):
    """
    Test that the values in the tensor returned by
    get_attention_to_amino_acid() for one amino acid are equal to the sum of
    the values in the columns of the input tensors corresponding to the same
    amino acid, for each layer and head.

    GIVEN: a list of torch.Tensor, a list of integers, and two integers
    WHEN: I call get_attention_to_amino_acid()
    THEN: the values in the tensor returned are equal to the sum of the values
        in the corresponding columns of the input tensors, for each layer and
        head

    """
    aa_pos_in_tokens = amino_acid_df.at[df_idx, "Position in Token List"]
    aa_cols = [
        [tensor[col].item() for col in aa_pos_in_tokens]
        for tensor in attention_column_sums
    ]
    output = get_attention_to_amino_acid(
        attention_column_sums,
        aa_pos_in_tokens,
        n_heads,
        n_layers,
    )

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            assert output[layer_idx, head_idx] == pytest.approx(
                sum(aa_cols[head_idx+layer_idx*n_heads])
            )


@pytest.mark.include_att_to_missing_aa
def test_include_att_to_missing_aa_returns_tensor(amino_acid_df, L_att_to_aa):
    """
    Test that include_att_to_missing_aa() returns a torch.Tensor.

    GIVEN: a pd.DataFrame and a list of torch.Tensor
    WHEN: I call include_att_to_missing_aa()
    THEN: the function returns a torch.Tensor

    """
    output = include_att_to_missing_aa(amino_acid_df, L_att_to_aa)

    assert isinstance(output, torch.Tensor)


@pytest.mark.include_att_to_missing_aa
def test_att_to_aa_shape(amino_acid_df, L_att_to_aa):
    """
    Test that the tensor returned by include_att_to_missing_aa() has shape
    (len(all_amino_acids), n_layers, n_heads).

    GIVEN: a pd.DataFrame, a list of torch.Tensor, and a list of strings with
        all the possible amino acids
    WHEN: I call include_att_to_missing_aa()
    THEN: the tensor returned has shape (len(all_amino_acids), n_layers,
        n_heads)

    """
    output = include_att_to_missing_aa(amino_acid_df, L_att_to_aa)

    assert output.shape == (len(all_amino_acids), 2, 3)


@pytest.mark.include_att_to_missing_aa
def test_att_to_aa_is_left_unchanged_for_present_amino_acids(
    amino_acid_df, L_att_to_aa
):
    """
    Test that the values in the tensor returned by include_att_to_missing_aa()
    referring to the amino acids present in the list of tokens are equal to the
    corresponding values in the list of torch.Tensor.

    GIVEN: a pd.DataFrame, a list of torch.Tensor, and a list of strings with
        all the possible amino acids
    WHEN: I call include_att_to_missing_aa()
    THEN: the values in the tensor returned referring to the amino acids
        present in the list of tokens are equal to the corresponding values in
        the list of torch.Tensor

    """
    output = include_att_to_missing_aa(amino_acid_df, L_att_to_aa)

    for idx_in, aa in enumerate(amino_acid_df["Amino Acid"]):
        idx_out = all_amino_acids.index(aa)
        assert torch.equal(output[idx_out], L_att_to_aa[idx_in])


@pytest.mark.include_att_to_missing_aa
def test_att_to_aa_is_zero_for_missing_amino_acids(amino_acid_df, L_att_to_aa):
    """
    Test that the values in the tensor returned by include_att_to_missing_aa()
    referring to amino acids not present in the list of tokens are all zero.

    GIVEN: a pd.DataFrame and a list of torch.Tensor
    WHEN: I call include_att_to_missing_aa()
    THEN: the values in the tensor returned referring to amino acids not
        present in the list of tokens are all zero

    """
    output = include_att_to_missing_aa(amino_acid_df, L_att_to_aa)

    for idx_out, aa in enumerate(all_amino_acids):
        if aa not in amino_acid_df["Amino Acid"].tolist():
            assert torch.equal(output[idx_out], torch.zeros(2, 3))


@pytest.mark.sum_attention_on_columns
def test_attention_on_columns_len(tuple_of_tensors):
    """
    Test that the list returned by sum_attention_on_columns() has length
    len(tuple_of_tensors)*(tensor.shape[-3]). In terms of attention, the list
    must have length equal to (n_layers*n_heads). This means that the tensors
    are unsqueezed along one dimension and stacked in a list.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call sum_attention_on_columns()
    THEN: the list returned has length len(tuple_of_tensors)*(tensor.shape[-3])

    """
    output = sum_attention_on_columns(tuple_of_tensors)

    assert all(
        len(output) == len(tuple_of_tensors)*tensor.shape[-3]
        for tensor in tuple_of_tensors
    )


@pytest.mark.sum_attention_on_columns
def test_attention_on_columns_shape(tuple_of_tensors):
    """
    Test that the tensors returned by sum_attention_on_columns() have shape
    equal to tensor.shape[-2] and tensor.shape[-1] -- the tensors are made to
    contain square matrices, therefore tensor.shape[-2] == tensor.shape[-1].
    This means that the square matrices are flattened along the columns. In
    terms of attention, the tensors must have shape equal to the number of
    residues in the chain.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call sum_attention_on_columns()
    THEN: the tensors in the list returned have shape equal to tensor.shape[-2]
        and tensor.shape[-1]

    """
    output = sum_attention_on_columns(tuple_of_tensors)

    for t_idx, t in enumerate(tuple_of_tensors):
        unsqueezed_dim = t.shape[-3]
        for dim_idx in range(unsqueezed_dim):
            assert output[dim_idx+t_idx*unsqueezed_dim].shape[0] == t.shape[-2]
            assert output[dim_idx+t_idx*unsqueezed_dim].shape[0] == t.shape[-1]


@pytest.mark.sum_attention_on_columns
def test_sum_over_columns(tuple_of_tensors):
    """
    Test that the values in the tensors returned by sum_attention_on_columns()
    are equal to the sum of the values in the corresponding columns of the
    input tensors.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call sum_attention_on_columns()
    THEN: the values in the tensors returned are equal to the sum of the values
        in the corresponding columns of the input tensors

    """
    output = sum_attention_on_columns(tuple_of_tensors)

    for t_idx, t in enumerate(tuple_of_tensors):
        unsqueezed_dim = t.shape[-3]
        """flattening not beyond the third to last dimension makes the test
        valid both for tensors with shape (batch_size, n_heads, seq_len,
        seq_len) -- e.g., in the case of attention just taken from ProtBert
        output -- and for tensors with shape (n_heads, seq_len, seq_len) -- as
        in the case of attention tensors returned by clean_attention()
        """
        t = torch.flatten(t, end_dim=-3)
        for dim_idx in range(unsqueezed_dim):
            assert all(
                output[dim_idx+t_idx*unsqueezed_dim][col_idx] == pytest.approx(
                    torch.sum(t[dim_idx, :, col_idx])
                )
                for col_idx in range(t.shape[-1])
            )


@pytest.mark.sum_attention_on_heads
def test_sum_attention_on_heads_returns_tensor(tuple_of_tensors):
    """
    Test that sum_attention_on_heads() returns a torch.Tensor.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call sum_attention_on_heads()
    THEN: the function returns a torch.Tensor

    """
    output = sum_attention_on_heads(tuple_of_tensors)

    assert isinstance(output, torch.Tensor)


@pytest.mark.sum_attention_on_heads
def test_attention_on_heads_shape(tuple_of_tensors):
    """
    Test that the tensor returned by sum_attention_on_heads() has shape
    (len(tuple_of_tensors), tensor.shape[-3]). This means that each tensor in
    the tuple is reduced to one number. In terms of attention, the tensors must
    have shape (n_layers, n_heads).

    GIVEN: a tuple of torch.Tensor
    WHEN: I call sum_attention_on_heads()
    THEN: the tensor returned has shape (len(tuple_of_tensors),
        tensor.shape[-3])

    """
    output = sum_attention_on_heads(tuple_of_tensors)

    assert output.shape[0] == len(tuple_of_tensors)
    assert all(output.shape[1] == t.shape[-3] for t in tuple_of_tensors)


@pytest.mark.sum_attention_on_heads
def test_sum_over_heads(tuple_of_tensors):
    """
    Test that the values in the tensor returned by sum_attention_on_heads() are
    equal to the sum of the values in the corresponding heads of the input
    tensors.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call sum_attention_on_heads()
    THEN: the values in the tensor returned are equal to the sum of the values
        in the corresponding heads of the input tensors

    """
    output = sum_attention_on_heads(tuple_of_tensors)

    for t_idx, t in enumerate(tuple_of_tensors):
        """flattening not beyond the third to last dimension makes the test
        valid both for tensors with shape (batch_size, n_heads, seq_len,
        seq_len) -- e.g., in the case of attention just taken from ProtBert
        output -- and for tensors with shape (n_heads, seq_len, seq_len) -- as
        in the case of attention tensors returned by clean_attention()
        """
        t = torch.flatten(t, end_dim=-3)
        for dim_idx in range(t.shape[-3]):
            assert output[t_idx, dim_idx] == pytest.approx(
                torch.sum(t[dim_idx])
            )


@pytest.mark.threshold_attention
def test_threshold_attention_returns_tuple_of_tensors(tuple_of_tensors):
    """
    Test that threshold_attention() returns a tuple of torch.Tensor.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call threshold_attention()
    THEN: the function returns a tuple of torch.Tensor

    """
    output = threshold_attention(tuple_of_tensors, 0.5)

    assert isinstance(output, tuple)
    assert all(isinstance(tensor, torch.Tensor) for tensor in output)


@pytest.mark.threshold_attention
def test_threshold_attention_leaves_shape_unchanged(tuple_of_tensors):
    """
    Test that the tensors returned by threshold_attention() have the same shape
    as the input tensors.

    GIVEN: a tuple of torch.Tensor
    WHEN: I call threshold_attention()
    THEN: the tensors returned have the same shape as the input tensors

    """
    output = threshold_attention(tuple_of_tensors, 0.5)

    assert all(
        t_out.shape == t_in.shape
        for t_out, t_in in zip(output, tuple_of_tensors)
    )


@pytest.mark.threshold_attention
@pytest.mark.parametrize("cutoff", param_cutoff)
def test_threshold_set_to_zero_values_below_cutoff(cutoff, tuple_of_tensors):
    """
    Test that the values in tuple_of_tensors that are below the cutoff are set
    to zero in the tensors returned by threshold_attention().

    GIVEN: a tuple of torch.Tensor
    WHEN: I call threshold_attention()
    THEN: the values below the cutoff are set to zero in the tensors returned

    """
    output = threshold_attention(tuple_of_tensors, cutoff)
    low_pass_input = [tensor < cutoff for tensor in tuple_of_tensors]

    assert all(
        torch.masked_select(t_out, t_in_bool) == pytest.approx(0.)
        for t_out, t_in_bool in zip(output, low_pass_input)
    )


@pytest.mark.threshold_attention
@pytest.mark.parametrize("cutoff", param_cutoff)
def test_threshold_leaves_unchanged_values_above_cutoff(
    cutoff, tuple_of_tensors
):
    """
    Test that the values in tuple_of_tensors that are above the cutoff are left
    unchanged in the tensors returned by threshold_attention().

    GIVEN: a tuple of torch.Tensor
    WHEN: I call threshold_attention()
    THEN: the values above the cutoff are left unchanged in the tensors
        returned

    """
    output = threshold_attention(tuple_of_tensors, cutoff)
    high_pass_input = [tensor >= cutoff for tensor in tuple_of_tensors]

    assert all(
        torch.masked_select(t_out, t_in_bool) == pytest.approx(
            torch.masked_select(t_in, t_in_bool)
        )
        for t_out, t_in_bool, t_in in zip(
            output, high_pass_input, tuple_of_tensors
        )
    )
