#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""__main__.py for ProtACon."""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from modules.attention import clean_attention
import preprocess_attention
import process_attention
import process_contact
import run_protbert
from IPython.display import display
from modules.utils import get_model_structure
import warnings


def main():
    """Run the scripts of ProtACon."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_attention, raw_tokens, CA_Atoms = run_protbert.main(seq_ID="1DVQ")

    number_of_heads, number_of_layers = get_model_structure(raw_attention)
    attention = clean_attention(raw_attention)
    tokens = raw_tokens[1:-1]
    type_of_amino_acids = list(dict.fromkeys(tokens))

    amino_acid_df, \
        attention_to_amino_acids, \
        percent_attention_to_amino_acids, \
        weighted_attention_to_amino_acids = preprocess_attention.main(
            attention, tokens)

    display(amino_acid_df)

    distance_map, norm_contact_map, binary_contact_map = process_contact.main(
        CA_Atoms)

    attention_sim_df, attention_per_layer, model_attention_average, \
        head_attention_alignment, layer_attention_alignment, \
        model_attention_alignment = process_attention.main(
            attention, attention_to_amino_acids, binary_contact_map,
            type_of_amino_acids)


if __name__ == '__main__':
    main()
