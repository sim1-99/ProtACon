#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ProtBert.

This script runs ProtBert on the peptide chain and returns the tokens and the
attention.
"""

from __future__ import annotations

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from typing import TYPE_CHECKING

import torch
from transformers import BertModel, BertTokenizer

from modules.miscellaneous import extract_CA_Atoms, get_sequence_to_tokenize
from modules.utils import read_pdb_file


if TYPE_CHECKING:
    from modules.miscellaneous import CA_Atom


def main(seq_ID: str) -> (tuple[torch.Tensor], list[str], tuple[CA_Atom]):
    """
    Run ProtBert on one peptide chain.

    The peptide chain is identified with its seq_ID. The function returns the
    tokens and the attention extracted from ProtBert.

    Parameters
    ----------
    seq_ID : str
        alphanumerical code representing uniquely one peptide chain

    Returns
    -------
    raw_attention : tuple[torch.Tensor]
        contains tensors that carry the attention from the model, including the
        attention relative to tokens [CLS] and [SEP]
    raw_tokens : list[str]
        contains strings which are the tokens used by the model, including the
        tokens [CLS] and [SEP]
    CA_Atoms: tuple[CA_Atom]

    """
    structure = read_pdb_file(seq_ID)
    CA_Atoms = extract_CA_Atoms(structure)
    sequence = get_sequence_to_tokenize(CA_Atoms)

    model_name = "Rostlab/prot_bert"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

    model = BertModel.from_pretrained(model_name, output_attentions=True)

    encoded_input = tokenizer.encode(sequence, return_tensors='pt')
    output = model(encoded_input)

    raw_tokens = tokenizer.convert_ids_to_tokens(encoded_input[0])
    raw_attention = output[-1]

    return raw_attention, raw_tokens, CA_Atoms
