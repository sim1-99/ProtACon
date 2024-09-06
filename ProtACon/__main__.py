#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""__main__.py file for command line application."""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

import argparse
import logging
from pathlib import Path

import torch

from ProtACon import config_parser
from ProtACon.modules.miscellaneous import load_model
from ProtACon.modules.utils import Loading, Timer
from ProtACon import align_with_contact


def parse_args():
    """Argument parser."""
    description = "ProtACon"
    parser = argparse.ArgumentParser(description=description)

    subparsers = parser.add_subparsers(
        dest="subparser",
        help="possible actions",
    )

    # on_set parser
    on_set = subparsers.add_parser(
        "on_set",
        help="get attention alignment and other quantities averaged over a set"
        " of peptide chains",
    )
    # optional arguments
    on_set.add_argument(
        "-s", "--save_single",
        action='store_true',
        help="save all plots relative to each single peptide chain",
    )
    on_set.add_argument(
        "-", "--proximity",  # add -
        action='store_true',
        help="get attention alignment respecting the contact map",
    )
    on_set.add_argument(
        "--instability",
        action='store_true',
        help="get attention alignment respecting the instability index",
    )
    on_set.add_argument(
        "-l", "--louvain",
        action='store_true',
        help="get attention alignment considering the communities found by louvain_partitions algorithm",
    )
    on_set.add_argument(
        "-k", "--kmeans",
        action='store_true',
        help="get attention alignment considering the clusters found by kmeans algorithm",
    )

    # on_chain parser
    on_chain = subparsers.add_parser(
        "on_chain",
        help="get attention alignment and other quantities for one single "
        "peptide chain",
    )
    # positional arguments
    on_chain.add_argument(
        "chain_code",
        type=str,
        help="code of the input peptide chain",
    )
    # visualization parser
    visualize = subparsers.add_parser(
        'net_work',
        help='specify if the tool to visualize the analisys on protein networks',
    )
    analysis_object = visualize.add_mutually_exclusive_group(
        required=True,
    )
    # require this
    analysis_object.add_argument(
        '-set', '--set_of_protein',
        action='store_true',
        help='config file to use for protein selection',
    )
    # or this one
    analysis_object.add_argument(
        '-code', '--single_protein_code',
        type=str,
        help='chain name to use for protein selection',
    )
    # positional
    visualize.add_argument(
        'plot_type',
        type=str,
        choices=['chain3D', 'pca', 'network'],
        help='type of plot to use for visualization',
    )
    # optional
    visualize.add_argument(
        '-r', '--print_results',
        action='store_true',
        help='decide if you want to visualize also the results of the analysis, '
        'both for the attention alignment, PCAs, and V-measure',
    )
    # positional
    visualize.add_argument(
        'analyse',
        type=str,
        choices=['louvain_community', 'kmeans', 'both', 'only_pca'],
        help='type of analysis to visualize results on',
    )
    # increase option
    visualize.add_argument(
        '-nc', '--node_color',
        type=str,
        default='ph_local',
        choices=['ph_local', 'ph_single', 'charge', 'flexy', 'iso_ph'],
        help='color of the node in the plots',
    )
    visualize.add_argument(
        '-ec', '--edge_color',
        type=str,
        default='instability',
        choices=['instability', 'contact_in_sequence', 'proximity'],
        help='color of the edge in the plots',
    )
    visualize.add_argument(
        '-es', '--edge_style',
        type=str,
        default='contact_in_sequence',
        choices=['instability', 'contact_in_sequence', 'proximity'],
    )
    visualize.add_argument(
        '-n', '--node_size',
        type=str,
        default='volume',
        choices=['volume', 'surface', 'radius_of_gyration', 'charge'],
        help='size of the node in the plots',
    )

    args = parser.parse_args()
    return args

    # type of graph to visualize: pca-plot / networkx / 3D viz
    # type of property: community_louvain /kmeans/features a color for cluster label also use the completness, homogeneity, v.measure as estimators
    # if features: contacts/sequence/proximity/feature of nodes: choice to be shown by dataframe,
    # define 2 feature: Analize or Vizualize to get or attention alignment or plot
    # ANALYZE : plot pca most important features + homogeneity, completness,vmeasure of cluster respecting web_grouping
    # VIZUALIZE: plot network/3D/pca plot of chain with properties associated
    # RESULTS: attention alignment of cluster
7


def main():
    """Run the script chosen by the user."""
    args = parse_args()

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    config = config_parser.Config("config.txt")

    paths = config.get_paths()
    plot_folder = paths["PLOT_FOLDER"]
    plot_dir = Path(__file__).resolve().parents[1]/plot_folder

    model_name = "Rostlab/prot_bert"
    with Loading("Loading the model"):
        model, tokenizer = load_model(model_name)

    if args.subparser == "on_set":
        proteins = config.get_proteins()
        protein_codes = proteins["PROTEIN_CODES"].split(" ")

        att_sim_df_list = []
        head_att_align_list = []
        layer_att_align_list = []

        with Timer("Total running time"):
            for code_idx, code in enumerate(protein_codes):
                with Timer(f"Running time for {code}"):
                    logging.info(f"Protein n.{code_idx+1}: {code}")
                    with torch.no_grad():
                        if args.save_single:
                            att_sim_df, head_att_align, layer_att_align = \
                                align_with_contact.main(code, args.save_single)
                        else:
                            att_sim_df, head_att_align, layer_att_align = \
                                align_with_contact.main(code)

                        att_sim_df_list.append(att_sim_df)
                        head_att_align_list.append(head_att_align)
                        layer_att_align_list.append(layer_att_align)

            avg_att_sim_df, avg_head_att_align, avg_layer_att_align = \
                align_with_contact.average_on_set(
                    att_sim_df_list, head_att_align_list, layer_att_align_list)

            align_with_contact.plot_average_on_set(
                avg_att_sim_df, avg_head_att_align, avg_layer_att_align)

    if (args.subparser == "on_chain" or args.subparser == "net_viz"):
        seq_ID = args.chain_code
        seq_dir = plot_dir/seq_ID
        seq_dir.mkdir(parents=True, exist_ok=True)

    if args.subparser == "on_chain":
        with Timer(f"Running time for {args.chain_code}"):
            with torch.no_grad():
                att_sim_df, head_att_align, layer_att_align = \
                    align_with_contact.main(args.chain_code, True)
    if args.subparser == 'net_work':
        analysis = ''
        if args.analyse == 'louvain_community':
            analysis = 'louvain'
        elif args.analyse == 'kmeans':
            analysis = 'km'
        elif args.analyse == 'both':
            analysis = 'both'
        elif args.analyse == 'only_pca':
            analysis = 'pca'

        if args.plot_type == 'chain3D':
            # use analysis
            pass
        elif args.plot_type == 'pca':
            # use analysis
            pass
        elif args.plot_type == 'network':
            # use analisys
            pass
        if args.print_results:
            # add result to be printed both in attention alignment, pca and v_measures
            pass


if __name__ == '__main__':
    main()
