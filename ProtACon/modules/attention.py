"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Define the functions for the extraction and processing of attention from the
ProtBert model.

"""
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import torch

from ProtACon.modules.basics import (
    all_amino_acids,
    get_model_structure,
)


def average_matrices_together(
    attention: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Average together the attention matrices independently for each layer.

    First, average together the attention matrices independently for each
    layer, which are stored in attention_per_layer. Then, average together the
    attention matrices in attention_per_layer, and store them in
    model_attention_average.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention retuned by the model, cleared of the attention related to
        the tokens [CLS] and [SEP].

    Returns
    -------
    attention_avgs : tuple[torch.Tensor, ...]
        The averages of the attention masks independently computed for each
        layer and, as last element, the average of those averages.

    Raises
    ------
    ValueError
        If the input tensors have a number of dimensions different from 3 or 4.

    """
    _, n_layers = get_model_structure(attention)

    attention_per_layer = [torch.empty(0) for _ in range(n_layers)]
    for layer_idx, layer in enumerate(attention):
        if len(layer.shape) == 4:  # if batch dimension is present
            layer = layer[0]
        elif len(layer.shape) == 3:
            pass
        else:
            raise ValueError("The input tensors must have 3 or 4 dimensions.")
        attention_per_layer[layer_idx] = \
            torch.sum(layer, dim=0)/layer.size(dim=0)
    model_attention_average = \
        torch.sum(torch.stack(attention_per_layer), dim=0)/n_layers

    attention_per_layer.append(model_attention_average)
    attention_avgs = tuple(attention_per_layer)

    return attention_avgs


def clean_attention(
    attention: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Remove the part of attention relative to non-amino acid tokens.

    If a tensor has a batch dimension, get rid of it -- (n_heads, seq_len,
    seq_len) instead of (1, n_heads, seq_len+2, seq_len+2) --, otherwise
    just remove the first and last rows and columns.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention matrices returned by the model, including the attention
        related to the tokens [CLS] and [SEP].

    Returns
    -------
    attention : tuple[torch.Tensor, ...]
        The attention returned by the model, cleared of the attention related
        to the tokens [CLS] and [SEP].

    Raises
    ------
    ValueError
        If the input tensors have a number of dimensions different from 3 or 4.

    """
    # "L_" stands for list
    L_attention = []
    for layer in attention:
        list_of_heads = []
        if len(layer.shape) == 4:  # if batch dimension is present
            layer = layer[0]
        elif len(layer.shape) == 3:
            pass
        else:
            raise ValueError("The input tensors must have 3 or 4 dimensions.")
        for head in layer:
            list_of_heads.append(head[1:-1, 1:-1])
        L_attention.append(torch.stack(list_of_heads))
    attention = tuple(L_attention)

    return attention


def compute_att_align_on_heads(
    attention: tuple[torch.Tensor, ...],
    indicator_function: np.ndarray,
) -> np.ndarray:
    """Compute the attention alignment on the attention heads of the model.

    The alignment is computed according to a property, represented with the
    binary map indicator_function. The tensors in the input tuple are the
    attention matrices in each layer -- with or without the batch dimension.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention matrices returned by the model.
    indicator_function : np.ndarray
        Binary map with shape (len(tokens), len(tokens)), representing one
        property of the peptide chain -- returns 1 if the property is present,
        0 otherwise.

    Returns
    -------
    attention_alignment : np.ndarray
        Array with shape (n_layers, n_heads), storing the portion of attention
        that aligns with the indicator_function.

    Raises
    ------
    ValueError
        If the input tensors have a number of dimensions different from 3
        or 4.

    """
    tensor_dims = [len(tensor.shape) for tensor in attention]

    # tensors are the attention matrices in each layer
    if all(dims in (3, 4) for dims in tensor_dims):
        n_heads, n_layers = get_model_structure(attention)
        attention_alignment = np.empty((n_layers, n_heads))
        for layer_idx, layer in enumerate(attention):
            # get rid of the batch dimension, if it is present
            layer = torch.flatten(layer, end_dim=-3)
            for head_idx, head in enumerate(layer):
                head = head.numpy()
                with np.errstate(all='raise'):
                    try:
                        attention_alignment[layer_idx, head_idx] = \
                            np.sum(head*indicator_function)/np.sum(head)
                    except FloatingPointError:
                        attention_alignment[layer_idx, head_idx] = \
                            np.float64(0)
    else:
        raise ValueError("The input tensors must have 3 or 4 dimensions.")

    return attention_alignment


def compute_att_align_on_layers(
    attention: tuple[torch.Tensor, ...],
    indicator_function: np.ndarray,
) -> np.ndarray:
    """Compute the attention alignment of the model layers with a property.

    The alignment is computed according to a property, represented with the
    binary map indicator_function. The tensors in the input tuple are the
    averages of the attention matrices in each layer.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The averages of the attention matrices in each layer.
    indicator_function : np.ndarray
        Binary map with shape (len(tokens), len(tokens)), representing one
        property of the peptide chain -- returns 1 if the property is present,
        0 otherwise.

    Returns
    -------
    attention_alignment : np.ndarray
        Array with shape (n_layers), storing the portion of attention that
        aligns with the indicator_function.

    Raises
    ------
    ValueError
        If the input tensors have a number of dimensions different from 2.

    """
    tensor_dims = [len(tensor.shape) for tensor in attention]

    # the tensors are the averages of the attention matrices in each layer
    if all(dims == 2 for dims in tensor_dims):
        n_layers = len(attention)
        attention_alignment = np.empty(n_layers)
        for layer_idx, layer in enumerate(attention):
            layer = layer.numpy()
            with np.errstate(all='raise'):
                try:
                    attention_alignment[layer_idx] = \
                        np.sum(layer*indicator_function)/np.sum(layer)
                except FloatingPointError:
                    attention_alignment[layer_idx] = np.float64(0)
    else:
        raise ValueError("The input tensors must have 2 dimensions.")

    return attention_alignment


def compute_attention_similarity(
    att_to_am_ac: torch.Tensor,
    am_ac: list[str],
) -> pd.DataFrame:
    """Assess the similarity of the attention between couples of amino acids.

    This is achieved by computing the Pearson correlation between the portion
    of attention that each amino acid receives across the heads. The diagonal
    obviously returns a perfect correlation (because the attention similarity
    between one amino acid and itself is total). Therefore, it is set to None.

    Parameters
    ----------
    att_to_am_ac : torch.Tensor
        Tensor with shape (len(am_ac), n_layers, n_heads), storing the
        attention given to each amino acid by each attention head.
    am_ac : list[str]
        The single letter codes of the amino acids.

    Returns
    -------
    att_sim_df : pd.DataFrame
        The attention similarity between each couple of amino acids.

    """
    n_heads = att_to_am_ac.shape[2]
    n_layers = att_to_am_ac.shape[1]

    att_sim_df = pd.DataFrame(data=None, index=am_ac, columns=am_ac)
    att_sim_df = att_sim_df[att_sim_df.columns].astype(float)

    for matrix1_idx, matrix1 in enumerate(att_to_am_ac):
        matrix1 = matrix1.reshape((n_heads*n_layers, ))
        for matrix2_idx, matrix2 in enumerate(att_to_am_ac):
            matrix2 = matrix2.reshape((n_heads*n_layers, ))

            corr = pearsonr(matrix1, matrix2)[0]
            att_sim_df.at[am_ac[matrix1_idx], am_ac[matrix2_idx]] = corr

            if matrix1_idx == matrix2_idx:
                att_sim_df.at[am_ac[matrix1_idx], am_ac[matrix2_idx]] = None

    return att_sim_df


def get_amino_acid_pos(
    amino_acid: str,
    tokens: list[str],
) -> list[int]:
    """Return the positions of a given amino acid along the list of tokens.

    Parameters
    ----------
    amino_acid : str
        The single letter amino acid code.
    tokens : list[str]
        The complete list of amino acid tokens.

    Returns
    -------
    amino_acid_pos : list[int]
        The positions of the tokens corresponding to amino_acid along the list
        tokens.

    """
    amino_acid_pos = [
        idx for idx, token in enumerate(tokens) if token == amino_acid
    ]

    return amino_acid_pos


def get_attention_to_amino_acid(
    att_column_sum: list[torch.Tensor],
    amino_acid_pos: list[int],
    n_heads: int,
    n_layers: int,
) -> torch.Tensor:
    """Compute the attention given from each attention head to one amino acid.

    Parameters
    ----------
    att_column_sum : list[torch.Tensor]
        (n_layers*n_heads) tensors, each with a length equal to the number of
        tokens, resulting from the column-wise sum over the attention values of
        each attention matrix.
    amino_acid_pos : list[int]
        The positions of the tokens corresponding to one amino acid along the
        list of tokens.
    n_heads : int
        The number of attention heads of the model.
    n_layers : int
        The number of layers of the model.

    Returns
    -------
    att_to_am_ac : torch.Tensor
        Tensor with shape (n_layers, n_heads), storing the attention given to
        each amino acid by each attention head.

    """
    # create an empty list; "L_" stands for list
    L_att_to_am_ac = [torch.empty(0) for _ in range(len(att_column_sum))]

    """ collect the values of attention given to one token by each head, then
    do the same with the next token representing the same amino acid
    """
    for head_idx, head in enumerate(att_column_sum):
        L_att_to_am_ac[head_idx] = head[amino_acid_pos[0]]
        for token_idx in range(1, len(amino_acid_pos)):
            """ since in each mask more than one column refer to the same amino
            acid, here we sum together all the "columns of attention" relative
            to the same amino acid
            """
            L_att_to_am_ac[head_idx] = torch.add(
                L_att_to_am_ac[head_idx],
                head[amino_acid_pos[token_idx]],
            )

    att_to_am_ac = torch.stack(L_att_to_am_ac).reshape(n_layers, n_heads)

    return att_to_am_ac


def include_att_to_missing_aa(
    amino_acid_df: pd.DataFrame,
    L_att_to_am_ac: list[torch.Tensor],
) -> torch.Tensor:
    """Insert zero matrices to represent the missing amino acids.

    Since the attention given to each amino acid is later used also for the
    attention analysis on more than one protein, it is necessary to fill the
    attention matrices relative to the missing amino acids with zeros. The
    items in L_att_to_am_ac are sorted by amino_acid_df["Amino Acid"] -- i.e.,
    alphabetically by amino acid -- but the data frame only includes the amino
    acids in the current chain. Therefore, I get the correspondence between the
    index of each amino acid in the data frame and the index of each amino acid
    in an alphabetically sorted list of all the possible amino acids. Finally,
    I fill a new list with the attention tensors in the right order -- that
    will be important later on for computing the attention similarity.

    Parameters
    ----------
    amino_acid_df : pd.DataFrame
        Data frame with the amino acids, the occurrences and the positions in
        the list of tokens of the residues in a chain.
    L_att_to_am_ac : list[torch.Tensor]
        Tensors with shape (n_layers, n_heads), each storing the attention
        given to one amino acid by each attention head.

    Returns
    -------
    torch.Tensor
        Tensor with shape (len(all_amino_acids), n_layers, n_heads), storing
        the attention given to each amino acid by each attention head.

    """
    n_heads = L_att_to_am_ac[0].shape[1]
    n_layers = L_att_to_am_ac[0].shape[0]
    L_att_to_all_am_ac = [
        torch.zeros(n_layers, n_heads) for _ in range(len(all_amino_acids))
    ]

    for old_idx in range(len(L_att_to_am_ac)):
        new_idx = amino_acid_df.at[old_idx, "Amino Acid"]
        new_idx = all_amino_acids.index(new_idx)
        L_att_to_all_am_ac[new_idx] = L_att_to_am_ac[old_idx]

    return torch.stack(L_att_to_all_am_ac)


def sum_attention_on_columns(
    attention: tuple[torch.Tensor, ...],
) -> list[torch.Tensor]:
    """Sum the values on the columns of each attention matrix.

    It returns a tensor for each attention matrix, each with length equal to
    the number of columns, that is, the number of tokens.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention matrices returned by the model.

    Returns
    -------
    att_column_sum : list[torch.Tensor]
        (n_layers*n_heads) tensors, each with a length equal to the number of
        tokens, resulting from the column-wise sum over the values of each
        attention matrix.

    Raises
    ------
    ValueError
        If the input tensors have a number of dimensions different from 3 or 4.

    """
    n_heads, n_layers = get_model_structure(attention)
    att_column_sum = [torch.empty(0) for _ in range(n_layers*n_heads)]

    for layer_idx, layer in enumerate(attention):
        if len(layer.shape) == 4:  # if batch dimension is present
            layer = layer[0]
        elif len(layer.shape) == 3:
            pass
        else:
            raise ValueError("The input tensors must have 3 or 4 dimensions.")
        for head_idx, head in enumerate(layer):
            att_column_sum[head_idx + layer_idx*n_heads] = torch.sum(head, 0)

    return att_column_sum


def sum_attention_on_heads(
    attention: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Sum all the values of each attention matrix.

    It returns a float number (the sum) for each attention matrix.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention matrices returned by the model.

    Returns
    -------
    att_head_sum : torch.Tensor
        Tensor with shape (n_layers, n_heads), resulting from the sum of all
        the values in each attention matrix.

    Raises
    ------
    ValueError
        If the input tensors have a number of dimensions different from 3 or 4.

    """
    n_heads, n_layers = get_model_structure(attention)
    att_head_sum = torch.zeros(n_layers, n_heads)

    for layer_idx, layer in enumerate(attention):
        if len(layer.shape) == 4:  # if batch dimension is present
            layer = layer[0]
        elif len(layer.shape) == 3:
            pass
        else:
            raise ValueError("The input tensors must have 3 or 4 dimensions.")
        for head_idx, head in enumerate(layer):
            att_head_sum[layer_idx, head_idx] = torch.sum(head).item()

    return att_head_sum


def threshold_attention(
    attention: tuple[torch.Tensor, ...],
    threshold: float,
) -> tuple[torch.Tensor, ...]:
    """Set to zero all the attention values below a given threshold.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention matrices returned by the model.
    threshold : float
        The threshold below which the attention values are set to zero.

    Returns
    -------
    thresholded : tuple[torch.Tensor, ...]
        The attention matrices, but all the values below the threshold are set
        to zero.

    """
    thresholded = tuple(
        [torch.where(tensor < threshold, 0., tensor) for tensor in attention]
    )

    return thresholded
