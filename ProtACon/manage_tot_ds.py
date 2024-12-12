"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-09-25

Create and operate with the data structures that store the sums of the
quantities computed for each peptide chain:

- the data frame with the amino acids in each peptide chain, storing the
  occurrences of each amino acid;
- the tensor to store the total values of the sums of all the values of
  attention in each head;
- the tensor to store the total values of the attention given to each amino
  acid;
- the array to store the total values of the attention alignment for each head;
- the array to store the total values of the attention alignment for each
  layer.

"""
import numpy as np
import pandas as pd
import torch

from ProtACon.modules.basics import all_amino_acids


def append_frequency_and_total(
    tot_amino_acid_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute and add to the data frame the percentage frequency of each amino
    acid over the whole set of proteins, and the total number of residues
    belonging to the considered amino acids.

    Parameters
    ----------
    tot_amino_acid_df : pd.DataFrame

    Returns
    -------
    tot_amino_acid_df : pd.DataFrame

    """
    tot_amino_acid_df.rename(
        columns={"Total AA Occurrences": "Occurrences"}, inplace=True
    )
    total_occurrences = tot_amino_acid_df["Occurrences"].sum()

    tot_amino_acid_df["Percentage Frequency (%)"] = \
        tot_amino_acid_df["Occurrences"]/total_occurrences*100

    tot_amino_acid_df["Total Occurrences"] = ""
    tot_amino_acid_df.at[0, "Total Occurrences"] = total_occurrences

    return tot_amino_acid_df


def create(
    n_layers: int,
    n_heads: int,
) -> tuple[
    pd.DataFrame,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    """
    Create the data structures to store the sum of the quantities computed for
    each peptide chain.

    Parameters
    ----------
    n_layers : int
        The number of layers in the model.
    n_heads : int
        The number of heads in the model.

    Returns
    -------
    tot_amino_acid_df : pd.DataFrame
        The data frame - with len(all_amino_acids) - to store the amino acids
        in each peptide chain and the occurrences of each of them.
    tot_att_head_sum : torch.Tensor
        The tensor - with shape (n_layers, n_heads) - to store the total values
        of the sums of all the values of attention in each head.
    tot_att_to_aa : torch.Tensor
        The tensor - with shape (len(all_amino_acids), n_layers, n_heads) - to
        store the total values of the attention given to each amino acid.
    tot_head_att_align : np.ndarray
        The array - with shape (n_layers, n_heads) - to store the total values
        of the attention alignment for each head.
    tot_layer_att_align : np.ndarray
        The array - with shape (n_layers) - to store the total values of the
        attention alignment for each layer.

    """
    tot_amino_acid_df = pd.DataFrame(
        data=0, index=all_amino_acids,
        columns=["Amino Acid", "Total AA Occurrences"]
    )
    tot_amino_acid_df["Amino Acid"] = all_amino_acids

    tot_att_head_sum = torch.zeros(n_layers, n_heads)
    tot_att_to_aa = torch.zeros(len(all_amino_acids), n_layers, n_heads)
    tot_head_att_align = np.zeros((n_layers, n_heads))
    tot_layer_att_align = np.zeros(n_layers)

    return (
        tot_amino_acid_df,
        tot_att_head_sum,
        tot_att_to_aa,
        tot_head_att_align,
        tot_layer_att_align,
    )


def keep_nonzero(
    tot_amino_acid_df: pd.DataFrame,
    tot_att_to_aa: torch.Tensor,
) -> tuple[
    pd.DataFrame,
    torch.Tensor,
]:
    """
    Drop the rows and the tensors relative to the amino acids with zero
    occurrences.

    Parameters
    ----------
    tot_amino_acid_df : pd.DataFrame
        The data frame - with len(all_amino_acids) - storing the amino acids in
        each peptide chain and the occurrences of each of them.
    tot_att_to_aa : torch.Tensor
        The tensor - with shape (len(all_amino_acids), n_layers, n_heads) -
        storing the total values of the attention given to each amino acid.

    Returns
    -------
    tot_amino_acid_df : pd.DataFrame
        The data frame - with len(chain_amino_acids) - storing the amino acids
        in each peptide chain and the occurrences of each of them.
    tot_att_to_aa : torch.Tensor
        The tensor - with shape (len(chain_amino_acids), n_layers, n_heads) -
        storing the total values of the attention given to each amino acid.

    """
    zero_indices = [
        idx for idx in range(len(tot_amino_acid_df)) if (
            tot_amino_acid_df.at[idx, "Occurrences"] == 0
        )
    ]
    nonzero_indices = [
        idx for idx in range(len(tot_amino_acid_df)) if (
            tot_amino_acid_df.at[idx, "Occurrences"] != 0
        )
    ]

    tot_amino_acid_df.drop(zero_indices, axis=0, inplace=True)
    tot_att_to_aa = torch.index_select(
        tot_att_to_aa, 0, torch.tensor(nonzero_indices)
    )

    return (
        tot_amino_acid_df,
        tot_att_to_aa,
    )


def update(
    tot_amino_acid_df: pd.DataFrame,
    tot_att_head_sum: torch.Tensor,
    tot_att_to_aa: torch.Tensor,
    tot_head_att_align: np.ndarray,
    tot_layer_att_align: np.ndarray,
    chain_ds: tuple[
        pd.DataFrame,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[
    pd.DataFrame,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    """
    Update the data structures storing the total values of the quantities, by
    summing the values computed for each peptide chain.

    Parameters
    ----------
    tot_amino_acid_df : pd.DataFrame
        The data frame - with len(all_amino_acids) - storing the amino acids in
        each peptide chain and the occurrences of each of them.
    tot_att_head_sum : torch.Tensor
        The tensor - with shape (n_layers, n_heads) - storing the total values
        of the sums of all the values of attention in each head.
    tot_att_to_aa : torch.Tensor
        The tensor - with shape (len(all_amino_acids), n_layers, n_heads) -
        storing the total values of the attention given to each amino acid.
    tot_head_att_align : np.ndarray
        The array - with shape (n_layers, n_heads) - storing the total values
        of the attention alignment for each head.
    tot_layer_att_align : np.ndarray
        The array - with shape (n_layers) - storing the total values of the
        attention alignment for each layer.
    chain_ds : tuple[pd.DataFrame, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
        amino_acid_df
            The data frame with the amino acids and the occurrences of each of
            them, for one peptide chain.
        att_head_sum
            Tensor with shape (n_layers, n_heads), resulting from the sum of
            all the values in each attention matrix, for one peptide chain.
        att_to_aa
            Tensor with shape (len(all_amino_acids), n_layers, n_heads),
            storing the attention given to each amino acid by each attention
            head, for one peptide chain.
        head_att_align
            Array with shape (n_layers, n_heads), storing how much attention
            aligns with indicator_function for each attention matrix, for one
            peptide chain.
        layer_att_align
            Array with shape (n_layers), storing how much attention aligns with
            indicator_function for each average attention matrix computed
            independently over each layer, for one peptide chain.

    Returns
    -------
    tot_amino_acid_df : pd.DataFrame
        The data frame - with len(all_amino_acids) - to store the amino acids
        in each peptide chain and the occurrences of each of them.
    tot_att_head_sum : torch.Tensor
        The tensor - with shape (n_layers, n_heads) - to store the total values
        of the sums of all the values of attention in each head.
    tot_att_to_aa : torch.Tensor
        The tensor - with shape (len(all_amino_acids), n_layers, n_heads) - to
        store the total values of the attention given to each amino acid.
    tot_head_att_align : np.ndarray
        The array - with shape (n_layers, n_heads) - to store the total values
        of the attention alignment for each head.
    tot_layer_att_align : np.ndarray
        The array - with shape (n_layers) - to store the total values of the
        attention alignment for each layer.

    """
    # in order to sum the data frames, we merge them...
    tot_amino_acid_df = pd.merge(
        tot_amino_acid_df, chain_ds[0][chain_ds[0].columns[:-2]],
        on="Amino Acid", how='left'
    )
    # ... then we sum the columns...
    tot_amino_acid_df["Total AA Occurrences"] = \
        tot_amino_acid_df["Occurrences"].add(
            tot_amino_acid_df["Total AA Occurrences"], fill_value=0
    )
    # ... and we drop the columns we don't need anymore
    tot_amino_acid_df.drop(columns=["Occurrences"], inplace=True)

    tot_att_head_sum = torch.add(tot_att_head_sum, chain_ds[1])
    tot_att_to_aa = torch.add(tot_att_to_aa, chain_ds[2])
    tot_head_att_align = np.add(tot_head_att_align, chain_ds[3])
    tot_layer_att_align = np.add(tot_layer_att_align, chain_ds[4])

    return (
        tot_amino_acid_df,
        tot_att_head_sum,
        tot_att_to_aa,
        tot_head_att_align,
        tot_layer_att_align,
    )
