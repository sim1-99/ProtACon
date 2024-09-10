"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Extract and pre-process the attention returned by ProtBert, by removing the
attention relative to the tokens [CLS] and [SEP]. Then, return the attention
given by the model to each amino acid.

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import warnings

from transformers import BertModel, BertTokenizer  # type: ignore
import pandas as pd
import torch

from ProtACon import config_parser
from ProtACon.modules.attention import (
    clean_attention,
    compute_weighted_attention,
    get_amino_acid_pos,
    get_attention_to_amino_acid,
    sum_attention_on_columns,
)
from ProtACon.modules.miscellaneous import (
    all_amino_acids,
    extract_CA_Atoms,
    get_model_structure,
    get_sequence_to_tokenize,
)
from ProtACon.modules.utils import (
    Logger,
    read_pdb_file,
)

if TYPE_CHECKING:
    from ProtACon.modules.miscellaneous import CA_Atom


log = Logger("cheesecake").get_logger()


def main(
    seq_ID: str,
    model: BertModel,
    tokenizer: BertTokenizer,
    save_opt: str,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[CA_Atom, ...],
    list[str],
    pd.DataFrame,
    list[torch.Tensor],
]:
    """
    Run ProtBert on one peptide chain and extract the attention. The peptide
    chain is identified with its seq_ID. Then, pre-process the attention.
    raw_attention and raw_tokens are cleared of the tokens [CLS] and [SEP].
    Then, the attention given to each amino acid is computed. It also contructs
    a data frame containing the information about the amino acids in the input
    peptide chain.

    Parameters
    ----------
    seq_ID : str
        The alphanumerical code representing uniquely the peptide chain.
    model : BertModel
    tokenizer : BertTokenizer
    save_opt : str
        One between ('none', 'plot', 'csv', 'both'). If 'csv' or 'both', save
        the amino acid data frame of every single chain.

    Returns
    -------
    attention : tuple[torch.Tensor, ...]
        The attention from the model, cleared of the attention relative to
        tokens [CLS] and [SEP].
    CA_Atoms : tuple[CA_Atom, ...]
    chain_amino_acids : list[str]
        The single letter codes of the amino acid types in the peptide chain.
    amino_acid_df : pd.DataFrame
        The data frame containing the information about the amino acids that
        constitute the peptide chain.
    list[torch.Tensor] :
        T_att_to_am_ac : torch.Tensor
            Tensor with shape (number_of_amino_acids, number_of_layers,
            number_of_heads), storing the absolute attention given to each
            amino acid by each attention head.
        T_rel_att_to_am_ac : torch.Tensor
            Tensor with shape (number_of_amino_acids, number_of_layers,
            number_of_heads), storing the relative attention given to each
            amino acid by each attention head; "rel" (relative) means that the
            values of attention given by one head to one amino acid are divided
            by the total value of attention of that head.
        T_weight_att_to_am_ac : torch.Tensor
            Tensor resulting from weighting T_rel_att_to_am_ac by the number of
            occurrences of the corresponding amino acid.

    """
    config = config_parser.Config("config.txt")

    paths = config.get_paths()
    file_folder = paths["FILE_FOLDER"]
    dfs_folder = "chain_dfs"

    dfs_dir = Path(__file__).resolve().parents[1]/file_folder/dfs_folder
    dfs_dir.mkdir(parents=True, exist_ok=True)

    save_if = ("csv", "both")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = read_pdb_file(seq_ID)
        CA_Atoms = extract_CA_Atoms(structure)
        sequence = get_sequence_to_tokenize(CA_Atoms)

        encoded_input = tokenizer.encode(sequence, return_tensors='pt')
        output = model(encoded_input)

    raw_tokens = tokenizer.convert_ids_to_tokens(encoded_input[0])
    raw_attention = output[-1]

    number_of_heads, number_of_layers = get_model_structure(raw_attention)

    attention = clean_attention(raw_attention)
    tokens = raw_tokens[1:-1]

    attention_on_columns = sum_attention_on_columns(attention)

    # remove duplicate amino acids from tokens and store the rest in a list
    chain_amino_acids = list(dict.fromkeys(tokens))

    # create two empty lists, having different lengths
    # "L_" stands for list
    L_att_to_am_ac = [
        torch.empty(0) for _ in range(len(chain_amino_acids))
    ]
    L_rel_att_to_am_ac = [
        torch.empty(0) for _ in range(len(chain_amino_acids))
    ]

    # start data frame construction
    columns = [
        "Amino Acid", "Occurrences", "Percentage Frequency (%)",
        "Position in Token List"
    ]
    amino_acid_df = pd.DataFrame(
        data=None, index=range(len(chain_amino_acids)), columns=columns
    )

    for amino_acid_idx, amino_acid in enumerate(chain_amino_acids):
        amino_acid_df.at[amino_acid_idx, "Amino Acid"] = amino_acid

        amino_acid_df.at[
            amino_acid_idx, "Position in Token List"
        ] = get_amino_acid_pos(amino_acid, tokens)

        amino_acid_df.at[
            amino_acid_idx, "Occurrences"
        ] = len(amino_acid_df.at[amino_acid_idx, "Position in Token List"])

        amino_acid_df.at[
            amino_acid_idx, "Percentage Frequency (%)"
        ] = amino_acid_df.at[amino_acid_idx, "Occurrences"]/len(tokens)*100

    # sort the residue types by alphabetical order
    amino_acid_df.sort_values(by=["Amino Acid"], inplace=True)
    # end data frame construction

    csv_file = dfs_dir/f"{seq_ID}_residue_df.csv"

    if csv_file.is_file() is False and save_opt in save_if:
        amino_acid_df.to_csv(
            csv_file, index=False, columns=columns, sep=';'
        )

    for idx in range(len(chain_amino_acids)):
        L_att_to_am_ac[idx], L_rel_att_to_am_ac[idx] = \
            get_attention_to_amino_acid(
                attention_on_columns,
                amino_acid_df.at[idx, "Position in Token List"],
                number_of_heads,
                number_of_layers,
            )

    L_weight_att_to_am_ac = compute_weighted_attention(
        L_rel_att_to_am_ac, amino_acid_df
    )

    # since rel_att_to_amino_acids and weight_att_to_amino_acids are later used
    # also for the attention analysis on more than one protein, it is necessary
    # to fill the attention matrices relative to the missing amino acids with
    # zeros
    missing_amino_acids = set(all_amino_acids) - set(chain_amino_acids)
    pos_missing_amino_acids = [
        all_amino_acids.index(am_ac) for am_ac in missing_amino_acids
    ]
    for pos in pos_missing_amino_acids:
        L_rel_att_to_am_ac.insert(
            pos, torch.zeros((number_of_layers, number_of_heads))
        )
        L_weight_att_to_am_ac.insert(
            pos, torch.zeros((number_of_layers, number_of_heads))
        )

    if (
        len(all_amino_acids) != len(L_rel_att_to_am_ac) or
        len(all_amino_acids) != len(L_weight_att_to_am_ac)
    ):
        raise ValueError(
            "The number of amino acids in the data frame is different from the"
            " number of amino acids in the attention tensors."
        )

    # "T_" stands for tensor
    T_att_to_am_ac = torch.stack(L_att_to_am_ac)
    T_rel_att_to_am_ac = torch.stack(L_rel_att_to_am_ac)
    T_weight_att_to_am_ac = torch.stack(L_weight_att_to_am_ac)

    log.logger.info(amino_acid_df)
    chain_amino_acids.sort()

    return (
        attention,
        CA_Atoms,
        chain_amino_acids,
        amino_acid_df,
        [
            T_att_to_am_ac,
            T_rel_att_to_am_ac,
            T_weight_att_to_am_ac,
        ],
    )
