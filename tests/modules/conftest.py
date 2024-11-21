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
def tuple_of_CA_Atom():
    """Tuple of CA_Atom objects."""
    return (
        CA_Atom(name="M", idx=5, coords=[0.0, -2.0, 11.0]),
        CA_Atom(name="L", idx=6, coords=[1.0, -1.0, 10.0]),
        CA_Atom(name="V", idx=7, coords=[2.0, 0.0, 9.0]),
    )


@pytest.fixture(scope="module")
def tuple_of_tensors():
    """Tuple of torch.Tensor."""
    return (
        torch.tensor(  # shape = (1, 2, 2, 2)
            [[[[0.1, 0.8], [0.5, 0.9]],
              [[0.2, 0.7], [0.6, 0.8]]]]
        ),
        torch.tensor(  # shape = (1, 2, 3, 3)
            [[[[0.5, 0.0, 0.6], [0.7, 0.7, 0.0], [0.1, 0.2, 0.3]],
              [[0.7, 0.4, 0.0], [0.8, 0.0, 0.0], [0.9, 0.6, 0.9]]]]
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
