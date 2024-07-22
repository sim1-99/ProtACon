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
    # type of graph to visualize: pca-plot / networkx / 3D viz
    #type of property: community_louvain /kmeans/features a color for cluster label also use the completness, homogeneity, v.measure as estimators
    # if features: contacts/sequence/proximity/feature of nodes: choice to be shown by dataframe,
    # define 2 feature: Analize or Vizualize to get or attention alignment or plot
# ANALYZE : plot pca most important features + homogeneity, completness,vmeasure of cluster respecting web_grouping
# VIZUALIZE: plot network/3D/pca plot of chain with properties associated
# RESULTS: attention alignment of cluster

    # 3d_viz parser
    net_viz = subparsers.add_parser(
        "net_viz",
        help="visualization of a protein displaying some selected properties "
        "or the attention alignment of a specific property",
        )
    # optional arguments
    net_parsers = net_viz.add_subparsers(
        dest="net_parser",
        help="possible actions to perfrom on network(s)",
    )
    # positional arguments
    net_viz.add_argument(
        "chain_code",
        type=str,
        help="code of the input peptide chain",
        )
    net_viz.add_argument(
        "property",
        type=str,
        help="property or network to show",
        )

    args = parser.parse_args()
    return args


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


if __name__ == '__main__':
    main()
