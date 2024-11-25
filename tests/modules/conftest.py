"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-14

Fixtures representing simple objects to test the correct functioning of the
functions in the modules.

"""
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
)
import pytest
import torch

from ProtACon.modules.basics import (
    CA_Atom,
)


@pytest.fixture(scope="module")
def chain_ID():
    """The PDB ID of a peptide chain."""
    return "2ONX"

@pytest.fixture(scope="module")
def tokens():
    """List of tokens."""
    return ["A", "M", "L", "V", "A", "Y", "D", "D"]


@pytest.fixture(scope="module")
def tuple_of_CA_Atom():
    """Tuple of CA_Atom objects."""
    return (
        CA_Atom(name="M", idx=5, coords=[0.0, -2.0, 11.0]),
        CA_Atom(name="L", idx=6, coords=[1.0, -1.0, 10.0]),
        CA_Atom(name="V", idx=7, coords=[2.0, 0.0, 9.0]),
    )


@pytest.fixture(scope="module")
def tuple_of_tensors():
    """
    Tuple of torch.Tensor.

    To simulate an attention matrix, every tensor must have the same number of
    heads (dim=1). The last two dimensions can vary from tensor to tensor, but
    they must be the same within the same tensor, and they cannot be less than
    3 -- taking into account tokens [CLS] and [SEP]. The first dimension
    represents the batch size, which is always 1 in the case of this pipeline.
    However, both the functions and their tests can work even with tensors
    without a batch dimension.

    """
    return (
        torch.tensor(  # shape = (1, 3, 4, 4)
            [[[[0.1, 0.8, 0.3, 0.6], [0.5, 0.9, 0.7, 0.7],
               [0.2, 0.4, 0.1, 0.0], [0.3, 0.5, 0.6, 0.8]],
              [[0.2, 0.7, 0.4, 0.5], [0.6, 0.8, 0.0, 0.0],
               [0.9, 0.1, 0.5, 0.2], [0.4, 0.3, 0.9, 0.7]],
              [[0.7, 0.9, 0.0, 0.5], [0.2, 0.0, 0.4, 0.1],
               [0.8, 0.3, 0.6, 0.9], [0.1, 0.6, 0.2, 0.3]]]]
        ),
        torch.tensor(  # shape = (3, 3, 3)
            [[[0.5, 0.0, 0.6], [0.0, 0.0, 0.0], [0.3, 0.2, 0.7]],
             [[0.7, 0.4, 0.0], [0.8, 0.0, 0.0], [0.9, 0.6, 0.9]],
             [[0.1, 0.0, 0.0], [0.2, 0.6, 0.0], [0.4, 0.3, 0.6]]]
        ),
    )


@pytest.fixture
def mocked_Bert(mocker):
    """Mock a BertModel and a BertTokenizer objects."""
    # skip a check about the existence of the vocab file in load_vocab()
    mocker.patch("os.path.isfile", return_value=True)
    # mock open() in load_vocab()
    mocker.patch("builtins.open", mocker.mock_open())

    mocker.patch(
        "transformers.BertModel.from_pretrained",
        return_value=BertModel(config=BertConfig()),
    )
    mocker.patch(
        "transformers.BertTokenizer.from_pretrained",
        return_value=BertTokenizer(
            vocab_file="mock_vocab.txt",
            clean_up_tokenization_spaces=True,
        ),
    )
