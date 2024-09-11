"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Define the functions for the extraction and processing of attention from the
ProtBert model.

"""
from scipy.stats import pearsonr  # type: ignore
import numpy as np
import pandas as pd
import torch

from ProtACon.modules.miscellaneous import get_model_structure


def average_matrices_together(
    attention: tuple[torch.Tensor, ...],
) -> list[torch.Tensor]:
    """
    First, average together the attention matrices independently for each
    layer, which are stored in attention_per_layer. Then, average together the
    attention matrices in attention_per_layer, and store them in
    model_attention_average.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention from the model, cleared of the attention relative to
        tokens [CLS] and [SEP].

    Returns
    -------
    attention_avgs : list[torch.Tensor]
        The averages of the attention masks independently computed for each
        layer and, as last element, the average of those averages.

    """
    _, number_of_layers = get_model_structure(attention)

    attention_per_layer = [torch.empty(0) for _ in range(number_of_layers)]
    for layer_idx, layer in enumerate(attention):
        attention_per_layer[layer_idx] = \
            torch.sum(layer, dim=0)/layer.size(dim=0)
    model_attention_average = \
        torch.sum(torch.stack(attention_per_layer), dim=0)/number_of_layers

    attention_per_layer.append(model_attention_average)
    attention_avgs = attention_per_layer

    return attention_avgs


def clean_attention(
    raw_attention: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """
    Remove from the attention the one relative to non-amino acid tokens.

    Parameters
    ----------
    raw_attention : tuple[torch.Tensor, ...]
        The attention from the model, including the attention relative to
        tokens [CLS] and [SEP].

    Returns
    -------
    T_attention: tuple[torch.Tensor, ...]
        The attention from the model, cleared of the attention relative to
        tokens [CLS] and [SEP].

    """
    # "L_" stands for list
    L_attention = []
    for layer_idx in range(len(raw_attention)):
        list_of_heads = []
        for head_idx in range(len(raw_attention[layer_idx][0])):
            list_of_heads.append(
                raw_attention[layer_idx][0][head_idx][1:-1, 1:-1]
            )
        L_attention.append(torch.stack(list_of_heads))
    attention = tuple(L_attention)

    return attention


def compute_attention_alignment(
    attention: tuple,
    indicator_function: np.ndarray,
) -> np.ndarray:
    """
    Compute the proportion of attention that aligns with a certain property.
    The property is represented with the binary map indicator_function.

    Parameters
    ----------
    attention : tuple
    indicator_function : np.ndarray
        The binary map representing one property of the peptide chain (returns
        1 if the property is present, 0 otherwise).

    Returns
    -------
    attention_alignment : np.ndarray
        The part of the attention that aligns with the indicator_function.

    """
    if len(attention[0].size()) == 2:
        number_of_layers = len(attention)
        attention_alignment = np.empty((number_of_layers))
        for layer_idx, layer in enumerate(attention):
            layer = layer.numpy()
            attention_alignment[layer_idx] = \
                np.sum(layer*indicator_function)/np.sum(layer)

    if len(attention[0].size()) == 3:
        number_of_heads, number_of_layers = get_model_structure(attention)
        attention_alignment = np.empty((number_of_layers, number_of_heads))
        for layer_idx, layer in enumerate(attention):
            for head_idx, head in enumerate(layer):
                head = head.numpy()
                attention_alignment[layer_idx, head_idx] = \
                    np.sum(head*indicator_function)/np.sum(head)

    return attention_alignment


def compute_attention_similarity(
    attention_to_amino_acids: torch.Tensor,
    chain_amino_acids: list[str],
) -> pd.DataFrame:
    """
    Assess the similarity of the attention received by each amino acids for
    each couple of amio acids. This is achieved by computing the Pearson
    correlation between the proportion of attention that each amino acid
    receives across heads. The diagonal obviously returns a perfect
    correlation (because the attention similarity between one amino acid and
    itself is total). Therefore, it is set to 0.

    Parameters
    ----------
    attention_to_amino_acids : torch.Tensor
        Tensor with shape (number_of_layers, number_of_heads), storing the
        attention (either absolute or relative or weighted) given to each
        amino acid by each attention head.
    chain_amino_acids : list[str]
        The single letter codes of the amino acid types in the peptide chain.

    Returns
    -------
    attention_sim_df : pd.DataFrame
        The attention similarity between each couple of amino acids.

    """
    number_of_heads = attention_to_amino_acids.shape[2]
    number_of_layers = attention_to_amino_acids.shape[1]

    attention_sim_df = pd.DataFrame(
        data=None, index=chain_amino_acids, columns=chain_amino_acids
    )
    attention_sim_df = attention_sim_df[attention_sim_df.columns].astype(float)

    for matrix1_idx, matrix1 in enumerate(attention_to_amino_acids):
        matrix1 = matrix1.numpy().reshape(
            (number_of_heads*number_of_layers, )
        )
        for matrix2_idx, matrix2 in enumerate(attention_to_amino_acids):
            matrix2 = matrix2.numpy().reshape(
                (number_of_heads*number_of_layers, )
            )
            corr = pearsonr(matrix1, matrix2)[0]
            attention_sim_df.at[
                chain_amino_acids[matrix1_idx],
                chain_amino_acids[matrix2_idx]
            ] = corr
            if matrix1_idx == matrix2_idx:
                attention_sim_df.at[
                    chain_amino_acids[matrix1_idx],
                    chain_amino_acids[matrix2_idx]
                ] = np.nan

    return attention_sim_df


def compute_weighted_attention(
    rel_att_to_amino_acids: list[torch.Tensor],
    amino_acid_df: pd.DataFrame,
) -> list[torch.Tensor]:
    """
    Compute the weighted attention given to each amino acid in the peptide
    chain. rel_att_to_amino_acids is weighted by the number of occurrences of
    each amino acid.

    Parameters
    ----------
    rel_att_to_amino_acids : torch.Tensor
        number_of_amino_acids tensors with shape (number_of_layers,
        number_of_heads), storing the relative attention in percentage given to
        each amino acid by each attention head; "rel" (relative) means that the
        values of attention given by one head to one amino acid are divided by
        the total value of attention of that head.
    amino_acid_df : pd.DataFrame
        The information about the amino acids in the input peptide chain.

    Returns
    -------
    weight_att_to_amino_acids : list[torch.Tensor]
        The tensors resulting from weighting rel_att_to_amino_acids by the
        number of occurrences of the corresponding amino acid.

    """
    weight_att_to_amino_acids = []
    occurrences = amino_acid_df["Occurrences"].tolist()

    for rel_att_to_amino_acid, occurrence in zip(
        rel_att_to_amino_acids, occurrences
    ):
        weight_att_to_amino_acids.append(rel_att_to_amino_acid/occurrence)

    return weight_att_to_amino_acids


def get_amino_acid_pos(
    amino_acid: str,
    tokens: list[str],
) -> list[int]:
    """
    Return the positions of a given token along the list of tokens.

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
    attention_on_columns: list[torch.Tensor],
    amino_acid_pos: list[int],
    number_of_heads: int,
    number_of_layers: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute the attention given from each attention head to each amino acid.
    The first tensor contains the absolute values of attention, while the
    second one contains the relative values. They take into account the fact 
    that we do not consider attention to tokens [CLS] and [SEP]. If those
    tokens are included, the sum of the attention values is the same in every
    head. This is no longer correct if the attention relative to those tokens
    is removed. Therefore, we have to correct possible distorsions that may
    rise as a consequence of that.

    Parameters
    ----------
    attention_on_columns : list[torch.Tensor]
        (number_of_layers*number_of_heads) tensors, each with a length equal to
        the number of tokens, resulting from the column-wise sum over the
        attention values of each attention matrix.
    amino_acid_pos : list[int]
        The positions of the tokens corresponding to one amino acid along the
        list of tokens.

    Returns
    -------
    T_att_to_am_ac : torch.Tensor
        Tensor with shape (number_of_layers, number_of_heads), storing the
        absolute attention given to each amino acid by each attention head.
    T_rel_att_to_am_ac : torch.Tensor
        Tensor with shape (number_of_layers, number_of_heads), storing the
        relative attention given to each amino acid by each attention head;
        "rel" (relative) means that the values of attention given by one head
        to one amino acid are divided by the total value of attention of that
        head.

    """
    # create two empty lists; "L_" stands for list
    L_att_to_am_ac = [
        torch.empty(0) for _ in range(len(attention_on_columns))
    ]
    L_rel_att_to_am_ac = [
        torch.empty(0) for _ in range(len(attention_on_columns))
    ]

    """ collect the values of attention given to one amino acid by each head,
    then do the same with the next amino acid
    """
    for head_idx, head in enumerate(attention_on_columns):
        L_att_to_am_ac[head_idx] = head[amino_acid_pos[0]]
        for amino_acid_idx in range(1, len(amino_acid_pos)):
            """ since in each mask more than one column refer to the same amino
            acid, here we sum together all the "columns of attention" relative
            to the same amino acid
            """
            L_att_to_am_ac[head_idx] = torch.add(
                L_att_to_am_ac[head_idx],
                head[amino_acid_pos[amino_acid_idx]]
            )

        """ here we compute the total value of attention of each mask, then
        we divide each value in L_att_to_am_ac by it
        """
        sum_over_head = torch.sum(head)
        L_rel_att_to_am_ac[head_idx] = L_att_to_am_ac[head_idx]/sum_over_head

    T_att_to_am_ac = torch.stack(L_att_to_am_ac)
    T_att_to_am_ac = torch.reshape(
        T_att_to_am_ac, (number_of_layers, number_of_heads)
    )

    T_rel_att_to_am_ac = torch.stack(L_rel_att_to_am_ac)
    T_rel_att_to_am_ac = torch.reshape(
        T_rel_att_to_am_ac, (number_of_layers, number_of_heads)
    )

    return (
        T_att_to_am_ac,
        T_rel_att_to_am_ac,
    )


'''def sum_attention(
    attention: tuple[torch.Tensor, ...],
) -> np.ndarray:
    """
    Sum all values of attention of each attention mask in a tuple of tensors.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention returned by the model.

    Returns
    -------
    attention_sum : np.ndarray
        Array with shape (number_of_layers, number_of_heads), resulting from
        the sum over all attention values of each attention mask.

    """
    number_of_heads = get_model_structure.number_of_heads
    number_of_layers = get_model_structure.number_of_layers
    attention_sum = np.zeros((number_of_layers, number_of_heads), dtype=float)

    for layer_idx, layer in enumerate(attention):
        for head_idx, head in enumerate(layer):
            attention_sum[layer_idx, head_idx] = float(torch.sum(head))

    return attention_sum
'''


def sum_attention_on_columns(
    attention: tuple[torch.Tensor, ...],
) -> list[torch.Tensor]:
    """
    Sum column-wise the values of attention of each mask in a tuple of tensors.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention returned by the model.

    Returns
    -------
    attention_on_columns : list[torch.Tensor]
        (number_of_layers*number_of_heads) tensors, each with a length equal to
        the number of tokens, resulting from the column-wise sum over the
        attention values of each attention matrix.

    """
    number_of_heads, number_of_layers = get_model_structure(attention)
    attention_on_columns = [
        torch.empty(0) for _ in range(number_of_layers*number_of_heads)
    ]

    for layer_idx, layer in enumerate(attention):
        for head_idx, head in enumerate(layer):
            attention_on_columns[head_idx + layer_idx*number_of_heads] = \
                torch.sum(head, 0)

    return attention_on_columns


def threshold_attention(
    attention: tuple[torch.Tensor, ...],
    threshold: float,
) -> tuple[torch.Tensor, ...]:
    """
    Set to zero all attention values below a certain threshold.

    Parameters
    ----------
    attention : tuple[torch.Tensor, ...]
        The attention returned by the model.
    threshold : float
        The threshold below which the attention values are set to zero.

    Returns
    -------
    thresholded : tuple[torch.Tensor, ...]
        The attention with all values below the threshold set to zero.

    """
    thresholded = []
    for tensor in attention:
        thresholded.append(torch.where(tensor < threshold, 0., tensor))

    return tuple(thresholded)
