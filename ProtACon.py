#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProtACon execution file."""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from IPython.display import display
import logging
from memory_profiler import profile
from pathlib import Path
import warnings

import config_parser
from modules.attention import clean_attention
from modules.miscellaneous import get_model_structure, get_types_of_amino_acids
from modules.plot_functions import plot_bars, plot_heatmap
from modules.utils import average_maps_together, Loading, Timer

import run_protbert
import preprocess_attention
import process_attention
import process_contact
import plotting

import numpy as np
import pandas as pd
import torch


@profile
def main(seq_ID: str) -> (torch.Tensor, pd.DataFrame, np.ndarray, np.ndarray):
    """
    Run the scripts of ProtACon for the peptide chain corresponding to seq_ID.

    Parameters
    ----------
    seq_ID : str
        alphanumerical code representing uniquely one peptide chain

    Returns
    -------
    weighted_attention_to_amino_acids : torch.Tensor
        tensor having dimension (number_of_amino_acids, number_of_layers,
        number_of_heads), resulting from weighting the relative attention in
        percentage given by each head to each amino acid by the number of
        occurrences of the corresponding amino acid
    attention_sim_df : pd.DataFrame
        stores attention similarity between each couple of amino acids
    head_attention_alignment : np.ndarray
        array having dimension (number_of_layers, number_of_heads), storing how
        much attention aligns with indicator_function for each attention masks
    layer_attention_alignment : np.ndarray
        array having dimension (number_of_layers), storing how much attention
        aligns with indicator_function for each average attention mask computed
        independently over each layer

    """
    global plot_dir
    # global attention_to_amino_acids_list
    global attention_sim_df_list
    global head_attention_alignment_list
    global layer_attention_alignment_list

    seq_dir = plot_dir/seq_ID
    seq_dir.mkdir(parents=True, exist_ok=True)

    with Loading("Loading the model"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_attention, raw_tokens, CA_Atoms = run_protbert.main(seq_ID)

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
    types_of_amino_acids.sort()

    distance_map, norm_contact_map, binary_contact_map = process_contact.main(
        CA_Atoms)

    attention_sim_df, \
        attention_per_layer, \
        model_attention_average, \
        head_attention_alignment, \
        layer_attention_alignment, \
        model_attention_alignment = process_attention.main(
            attention, attention_to_amino_acids, binary_contact_map,
            types_of_amino_acids)

    attention_to_amino_acids = (attention_to_amino_acids,
                                percent_attention_to_amino_acids,
                                weighted_attention_to_amino_acids)

    attention_alignment = (head_attention_alignment, layer_attention_alignment)

    plotting.main(
        distance_map, norm_contact_map, binary_contact_map, attention,
        attention_per_layer, model_attention_average, attention_to_amino_acids,
        attention_sim_df, attention_alignment, seq_dir, types_of_amino_acids)

    # attention_to_amino_acids_list.append(weighted_attention_to_amino_acids)
    attention_sim_df_list.append(attention_sim_df)
    head_attention_alignment_list.append(head_attention_alignment)
    layer_attention_alignment_list.append(layer_attention_alignment)


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    config = config_parser.Config("config.txt")

    proteins = config.get_proteins()
    protein_codes = proteins["PROTEIN_CODES"].split(" ")

    paths = config.get_paths()
    config_parser.ensure_storage_directories_exist(paths)
    plot_folder = paths["PLOT_FOLDER"]
    plot_dir = Path(__file__).parent/plot_folder

    # attention_to_amino_acids_list = []
    attention_sim_df_list = []
    head_attention_alignment_list = []
    layer_attention_alignment_list = []

    with Timer("Total running time"):
        for code_idx, code in enumerate(protein_codes):
            with Timer(f"Running time for {code}"):
                logging.info(f"Protein n.{code_idx+1}: {code}")
                with torch.no_grad():
                    main(code)

    """
    average_attention_to_amino_acids = average_maps_together(
        attention_to_amino_acids_list)
    logging.info("Plotting Average Attention to Amino Acids")
    plot_attention_to_amino_acids(
        average_attention_to_amino_acids,
        types_of_amino_acids,
        plot_title="Average Attention to Amino Acids")
    logging.info("Done")
    """
    with Loading("Carrying out average attention similarity"):
        average_attention_sim_df = average_maps_together(attention_sim_df_list)
        plot_heatmap(average_attention_sim_df,
                     plot_title="Average Pairwise Attention Similarity\n"
                     "Pearson Correlation")

    average_attention_sim_df.to_csv(
        plot_dir/"attention_sim_df.csv", index=True, sep=';')

    with Loading("Carrying out average head attention alignment"):
        average_head_attention_alignment = average_maps_together(
            head_attention_alignment_list)
        plot_heatmap(average_head_attention_alignment,
                     plot_title="Average Head Attention Alignment")

    with Loading("Carrying out average layer attention alignment"):
        average_layer_attention_alignment = average_maps_together(
            layer_attention_alignment_list)
        plot_bars(average_layer_attention_alignment,
                  plot_title="Average Layer Attention Alignment")
