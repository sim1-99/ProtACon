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
    threshold_attention,
)


pytestmark = pytest.mark.attention


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
    Having the input tensors shape (batch_size, n_heads, seq_len+2, seq_len+2),
    test that the tensors returned by clean_attention() have shape (n_heads,
    seq_len, seq_len).

    GIVEN: a tuple of torch.Tensor with shape (batch_size, n_heads, seq_len+2,
        seq_len+2)
    WHEN: I call clean_attention()
    THEN: the tensors in the tuple returned have shape (n_heads, seq_len,
        seq_len)

    """
    output = clean_attention(tuple_of_tensors)

    assert all(
        t1.shape[0] == t2.shape[1]
        for t1, t2 in zip(output, tuple_of_tensors)
    )
    assert all(
        t1.shape[1] == t2.shape[2]-2
        for t1, t2 in zip(output, tuple_of_tensors)
    )
    assert all(
        t1.shape[2] == t2.shape[3]-2
        for t1, t2 in zip(output, tuple_of_tensors)
    )


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

    assert all(
        torch.sum(output[i]) == pytest.approx(
            torch.sum(tuple_of_tensors[i][0, :, 1:-1, 1:-1])
        )
        for i in range(len(output))
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