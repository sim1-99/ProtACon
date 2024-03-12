#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProtACon execution file."""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

import config_parser
from modules.attention import clean_attention
from modules.utils import get_model_structure, get_types_of_amino_acids, Timer
import run_protbert
import preprocess_attention
import process_attention
import process_contact
import plotting

from IPython.display import display
import logging
from pathlib import Path
import warnings


def main(seq_ID: str):
    """Run the scripts of ProtACon."""
    global plot_dir
    seq_dir = plot_dir/seq_ID
    seq_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Load the model for {seq_ID}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_attention, raw_tokens, CA_Atoms = run_protbert.main(seq_ID)

    logging.info(f"Model for {seq_ID} has been loaded")

    attention = clean_attention(raw_attention)
    tokens = raw_tokens[1:-1]
    number_of_heads, number_of_layers = get_model_structure(raw_attention)
    types_of_amino_acids = get_types_of_amino_acids(tokens)

    amino_acid_df, \
        attention_to_amino_acids, \
        percent_attention_to_amino_acids, \
        weighted_attention_to_amino_acids = preprocess_attention.main(
            attention, tokens, seq_dir)

    display(amino_acid_df)

    distance_map, norm_contact_map, binary_contact_map = process_contact.main(
        CA_Atoms)

    attention_sim_df, \
        attention_per_layer, \
        model_attention_average, \
        head_attention_alignment, \
        layer_attention_alignment, \
        model_attention_alignment = process_attention.main(
            attention, attention_to_amino_acids, binary_contact_map)

    attention_to_amino_acids = (attention_to_amino_acids,
                                percent_attention_to_amino_acids,
                                weighted_attention_to_amino_acids)

    attention_alignment = (head_attention_alignment, layer_attention_alignment)

    plotting.main(
        distance_map, norm_contact_map, binary_contact_map, attention,
        attention_per_layer, model_attention_average, attention_to_amino_acids,
        attention_sim_df, attention_alignment, seq_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    config = config_parser.Config("config.txt")

    proteins = config.get_proteins()
    protein_codes = proteins['PROTEIN_CODES'].split(" ")

    paths = config.get_paths()
    config_parser.ensure_storage_directories_exist(paths)
    plot_folder = paths["PLOT_FOLDER"]
    plot_dir = Path(__file__).parent/plot_folder

    with Timer("Total running time"):
        for code_idx, code in enumerate(protein_codes):
            with Timer(f"Running time for {code}"):
                logging.info(f"Protein n.{code_idx+1}")
                main(code)
