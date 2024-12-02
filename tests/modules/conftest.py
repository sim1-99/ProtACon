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
import numpy as np
import pandas as pd
import pytest
import torch

from ProtACon.modules.basics import CA_Atom


@pytest.fixture(scope="module")
def amino_acid_df():
    """
    Data frame with the amino acids, the occurrences and the positions in the
    list of tokens of the residues in a chain.

    """
    data = {
        "Amino Acid": ["A", "M", "V"],
        "Occurrences": [2, 1, 1],
        "Percentage Frequency (%)": [50.0, 25.0, 25.0],
        "Position in Token List": [[0, 2], [1], [3]],
    }
    amino_acid_df = pd.DataFrame(data=data, index=range(3))

    return amino_acid_df


@pytest.fixture(scope="module")
def amino_acids_in_chain(tokens):
    """List of amino acids in a peptide chain."""
    return list(dict.fromkeys(tokens))


@pytest.fixture(scope="module")
def attention_column_sums():
    """
    List of tensors with the column-wise sums of the values of the attention
    matrices.

    """
    return [         # A    M    A    V      layer-head
        torch.tensor([1.5, 0.8, 1.3, 0.4]),  # 0-0
        torch.tensor([2.8, 0.1, 0.2, 0.9]),  # 0-1
        torch.tensor([1.4, 0.6, 1.4, 0.6]),  # 0-2
        torch.tensor([2.1, 0.8, 0.2, 0.9]),  # 1-0
        torch.tensor([0.3, 0.3, 1.5, 1.9]),  # 1-1
        torch.tensor([1.3, 0.6, 1.4, 0.7]),  # 1-2
    ]


@pytest.fixture(scope="module")
def chain_ID():
    """The PDB ID of a peptide chain."""
    return "2ONX"


@pytest.fixture(scope="module")
def array_2d():
    """
    2D numpy array.

    It can be used to simulate a distance map, whose entries are the euclidean
    distances between the alpha-carbon atoms of the residues in a peptide
    chain. In this case, the matrix must be square and symmetric, with the
    diagonal filled with zeros.

    """
    return np.array(
        [[0.0, 9.8, 6.2, 2.5],
         [9.8, 0.0, 5.0, 1.7],
         [6.2, 5.0, 0.0, 8.4],
         [2.5, 1.7, 8.4, 0.0]],
        dtype=float,
    )


@pytest.fixture(scope="module")
def bin_array_2d():
    """
    2D numpy array with binary values.

    It can be used to simulate a binary contact map, assigning 1 to the entries
    that represent the a couple of residues that are in contact, and 0 to the
    others. In this case, the matrix must be square and symmetric, with the
    diagonal filled with zeros.

    """
    return np.array(
        [[0, 1, 0, 1],
         [1, 0, 0, 1],
         [0, 0, 0, 1],
         [1, 1, 1, 0]],
        dtype=int,
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


@pytest.fixture(scope="module")
def L_att_to_aa():
    """
    List of tensors with the attention given to each amino acid by each
    attention head.

    """
    return [         # 0-0  0-1  0-2    1-0  1-1  1-2
        torch.tensor([[2.8, 3.0, 2.8], [2.3, 1.8, 2.7]]),  # amino acid 0 ("A")
        torch.tensor([[0.8, 0.1, 0.6], [0.8, 0.3, 0.6]]),  # amino acid 1 ("M")
        torch.tensor([[0.4, 0.9, 0.6], [0.9, 0.9, 0.7]]),  # amino acid 2 ("V")
    ]


@pytest.fixture(scope="module")
def n_heads():
    """Number of attention heads in a model."""
    return 3


@pytest.fixture(scope="module")
def n_layers():
    """Number of layers in a model."""
    return 2


@pytest.fixture(scope="module")
def tokens():
    """List of tokens of a peptide chain."""
    return ["A", "M", "A", "V"]


@pytest.fixture(scope="module")
def T_att_to_aa(L_att_to_aa):
    """
    Tensor with the attention given to each amino acid by each attention head.

    """
    return torch.stack(L_att_to_aa)  # shape = (4, 2, 3)


@pytest.fixture(scope="module")
def tuple_of_CA_Atom():
    """
    Tuple of the CA_Atom objects representing the alpha-carbon atom of each
    residue in a peptide chain.

    """
    return (
        CA_Atom(name="A", idx=0, coords=[1.0, -1.0, 10.0]),
        CA_Atom(name="M", idx=1, coords=[0.0, -2.0, 11.0]),
        CA_Atom(name="A", idx=2, coords=[6.0, 0.0, 10.0]),
        CA_Atom(name="V", idx=3, coords=[2.0, 0.0, 9.0]),
    )


@pytest.fixture(scope="module")
def tuple_of_tensors():
    """
    Tuple of torch.Tensor.

    To simulate an attention matrix, all the tensors in the tuple must have the
    same number of heads (dim -3), and the same number of entries in the square
    matrices (dim -2 and -1), that represent the number of tokens got from a
    peptide chain. The last two dimensions must be equal and cannot be less
    than 3 -- taking into account that the first and the last columns and rows
    refer to the tokens [CLS] and [SEP], that are discarded.

    The first dimension in the first tensor here below represents the batch
    size, which is always 1 in the case of this pipeline. However, both the
    functions and their tests can work even with tensors without a batch
    dimension. For this reason, the two tensors in this tuple have different
    shape, but this would not be the case in a real scenario.

    Each square matrix should also sum to the number of tokens -- including
    [CLS] and [SEP] -- and each row should sum to 1. This details is left out
    here though, as it is not necessary for the tests.

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
        torch.tensor(  # shape = (3, 4, 4)
            [[[0.5, 0.0, 0.6, 0.0], [0.0, 0.0, 0.3, 0.2],
              [0.5, 0.3, 0.1, 0.7], [0.9, 0.8, 0.4, 0.5]],
             [[0.7, 0.4, 0.0, 0.8], [0.6, 0.0, 0.9, 0.6],
              [0.9, 0.2, 0.5, 0.1], [0.3, 0.7, 0.8, 0.4]],
             [[0.1, 0.0, 0.0, 0.2], [0.6, 0.0, 0.4, 0.3],
              [0.6, 0.5, 0.8, 0.9], [0.7, 0.1, 0.7, 0.5]]]
        ),
    )
