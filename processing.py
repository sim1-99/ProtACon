#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processing."""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from modules.attention import get_sequence_to_tokenize
from modules.utils import extract_CA_Atoms, read_pdb_file
from transformers import BertModel, BertTokenizer


seq_ID = "1DVQ" """TODO: substitute with an element from a tuple of amino acids
                   defined in a __main__ script"""
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
del output

number_of_layers = len(raw_attention)
layer_structure = raw_attention[0].shape
number_of_heads = layer_structure[1]
