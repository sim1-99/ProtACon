#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""__main__.py for ProtACon."""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from IPython.display import display
from modules.utils import get_model_structure
import process_attention
import run_protbert
import warnings


def main():
    """Run the scripts of ProtACon."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_attention, raw_tokens = run_protbert.main(seq_ID="1DVQ")

    number_of_heads, number_of_layers = get_model_structure(raw_attention)

    amino_acid_df, \
        attention_to_amino_acids, \
        percent_attention_to_amino_acids, \
        weighted_attention_to_amino_acids = process_attention.main(
            raw_attention, raw_tokens)

    display(amino_acid_df)


if __name__ == '__main__':
    main()
