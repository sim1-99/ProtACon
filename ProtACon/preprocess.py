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

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from transformers import BertModel, BertTokenizer  # type: ignore
import pandas as pd
import torch

from ProtACon import config_parser
from ProtACon.modules.attention import (
    clean_attention,
    get_amino_acid_pos,
    get_attention_to_amino_acid,
    sum_attention_on_columns,
    sum_attention_on_heads,
    threshold_attention,
)
from ProtACon.modules.miscellaneous import (
    all_amino_acids,
    extract_CA_Atoms,
    get_model_structure,
    get_sequence_to_tokenize,
    read_pdb_file,
)
from ProtACon.modules.utils import Logger

if TYPE_CHECKING:
    from ProtACon.modules.miscellaneous import CA_Atom


log = Logger("mylog").get_logger()


def main(
    seq_ID: str,
    model: BertModel,
    tokenizer: BertTokenizer,
    save_opt: str,
) -> tuple[
    tuple[torch.Tensor, ...],
    torch.Tensor,
    tuple[CA_Atom, ...],
    pd.DataFrame,
    torch.Tensor,
]:
    """
    Run ProtBert on one peptide chain and extract the attention. The peptide
    chain is identified with its seq_ID. Then, pre-process the attention.
    raw_attention and raw_tokens are cleared of the tokens [CLS] and [SEP].
    Optionally, thresholding on attention is applied. Finally, the attention
    given to each amino acid is computed. It also contructs a data frame with
    the information about the amino acids in the input peptide chain.

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
    att_head_sum : torch.Tensor
        Tensor with shape (n_layers, n_heads), resulting from the sum of all
        the values in each attention matrix.
    CA_Atoms : tuple[CA_Atom, ...]
    amino_acid_df : pd.DataFrame
        The data frame containing the information about the amino acids that
        constitute the peptide chain.
    T_att_to_aa : torch.Tensor
        Tensor with shape (len(all_amino_acids), n_layers, n_heads), storing
        the attention given to each amino acid by each attention head.

    """
    config = config_parser.Config("config.txt")

    cutoffs = config.get_cutoffs()
    paths = config.get_paths()
    
    att_cutoff = cutoffs["ATTENTION_CUTOFF"]

    file_folder = paths["FILE_FOLDER"]
    dfs_folder = "chain_dfs"

    dfs_dir = Path(__file__).resolve().parents[1]/file_folder/dfs_folder
    dfs_dir.mkdir(parents=True, exist_ok=True)

    save_if = ("csv", "both")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        structure = read_pdb_file(seq_ID)
        CA_Atoms = extract_CA_Atoms(structure)
        sequence = get_sequence_to_tokenize(CA_Atoms)

        encoded_input = tokenizer.encode(sequence, return_tensors='pt')
        output = model(encoded_input)

    raw_tokens = tokenizer.convert_ids_to_tokens(encoded_input[0])
    raw_attention = output[-1]

    n_heads, n_layers = get_model_structure(raw_attention)

    tokens = raw_tokens[1:-1]

    cl_attention = clean_attention(raw_attention)
    attention = threshold_attention(cl_attention, att_cutoff)

    att_column_sum = sum_attention_on_columns(attention)
    att_head_sum = sum_attention_on_heads(attention)

    # remove duplicate amino acids from tokens and store the rest in a list
    chain_amino_acids = list(dict.fromkeys(tokens))

    # start data frame construction
    columns = [
        "Amino Acid", "Occurrences", "Percentage Frequency (%)",
        "Position in Token List"
    ]
    amino_acid_df = pd.DataFrame(
        data=None, index=range(len(chain_amino_acids)), columns=columns
    )

    for am_ac_idx, am_ac in enumerate(chain_amino_acids):
        amino_acid_df.at[am_ac_idx, "Amino Acid"] = am_ac

        amino_acid_df.at[am_ac_idx, "Position in Token List"] = \
            get_amino_acid_pos(am_ac, tokens)

        amino_acid_df.at[am_ac_idx, "Occurrences"] = \
            len(amino_acid_df.at[am_ac_idx, "Position in Token List"])

        amino_acid_df.at[am_ac_idx, "Percentage Frequency (%)"] = \
            amino_acid_df.at[am_ac_idx, "Occurrences"]/len(tokens)*100

    # sort the amino acids by alphabetical order
    amino_acid_df.sort_values(by=["Amino Acid"], inplace=True)

    csv_file = dfs_dir/f"{seq_ID}_residue_df.csv"
    if csv_file.is_file() is False and save_opt in save_if:
        amino_acid_df.to_csv(csv_file, index=False, columns=columns, sep=';')
    # end data frame construction and save it

    # create an empty list; "L_" stands for list
    L_att_to_aa = [torch.empty(0) for _ in range(len(chain_amino_acids))]

    for idx in range(len(chain_amino_acids)):
        L_att_to_aa[idx] = get_attention_to_amino_acid(
            att_column_sum,
            amino_acid_df.at[idx, "Position in Token List"],
            n_heads,
            n_layers,
        )

    """Since the attention given to each amino acid is later used also for the
    attention analysis on more than one protein, it is necessary to fill the
    attention matrices relative to the missing amino acids with zeros;
    L_att_to_all_am_ac is used for this purpose
    """
    L_att_to_all_aa = [
        torch.zeros(n_layers, n_heads) for _ in range(len(all_amino_acids))
    ]

    """The loop is because items in L_att_to_aa are not sorted by
    amino_acid_df["Amino Acid"] - i.e., alphabetically by amino acid - but by
    amino_acid_df.index. Therefore, I get the correspondence between the index
    of each amino acid in the data frame and the index of each amino acid in an
    alphabetically sorted list of all the possible amino acids. Finally, I fill
    a new list with the attention tensors in the right order - that is
    important later for the attention similarity.
    """
    for old_idx in range(len(L_att_to_aa)):
        new_idx = amino_acid_df.at[amino_acid_df.index[old_idx], "Amino Acid"]
        new_idx = all_amino_acids.index(new_idx)
        L_att_to_all_aa[new_idx] = L_att_to_aa[old_idx]

    assert len(all_amino_acids) == len(L_att_to_all_aa), (
        "The number of amino acids in the data frame must be equal to the"
        " number of amino acids in the attention tensors."
    )

    # "T_" stands for tensor
    T_att_to_aa = torch.stack(L_att_to_all_aa)

    log.logger.info(amino_acid_df)

    return (
        attention,
        att_head_sum,
        CA_Atoms,
        amino_acid_df,
        T_att_to_aa,
    )
