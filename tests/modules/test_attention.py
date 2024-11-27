"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-13

Test suite for the functions in preprocess.py.

"""
import pytest
import torch

from ProtACon.modules.attention import (
    clean_attention,
    get_amino_acid_pos,
    get_attention_to_amino_acid,
    include_att_to_missing_aa,
    sum_attention_on_columns,
    sum_attention_on_heads,
    threshold_attention,
)
from ProtACon.modules.basics import all_amino_acids


pytestmark = pytest.mark.attention
param_aa = ["A", "M", "L", "V", "Y", "D"]
param_cutoff = [0.0, 0.5, 0.9, 1.1]
param_df_idx = list(range(3))


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
