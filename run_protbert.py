#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ProtBert.

This script runs ProtBert on the peptide chain and returns the tokens and the
attention.
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from modules.utils import extract_CA_Atoms, get_sequence_to_tokenize, \
    read_pdb_file

from transformers import BertModel, BertTokenizer


def main(seq_ID: str) -> (tuple, list, tuple):
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
    raw_attention : tuple
        contains tensors that carry the attention from the model, including the
        attention relative to tokens [CLS] and [SEP]
    raw_tokens : list
        contains strings which are the tokens used by the model, including the
        tokens [CLS] and [SEP]
    CA_Atoms: tuple

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
    del output, sequence, structure

    return raw_attention, raw_tokens, CA_Atoms
