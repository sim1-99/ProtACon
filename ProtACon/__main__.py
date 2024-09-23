"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella, R. Eliasy

__main__.py file for command line application.

"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ProtACon import config_parser
from ProtACon.modules.miscellaneous import (
    all_amino_acids,
    fetch_pdb_entries,
    get_model_structure,
    load_model,
)
from ProtACon.modules.utils import (
    Logger,
    Loading,
    Timer,
)
from ProtACon import align_with_contact
from ProtACon import compute_on_set
from ProtACon import plotting
from ProtACon import preprocess


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
        help="get the attention alignment with one of the positional"
        " arguments for a set of peptide chains",
    )
    # positional arguments
    on_set.add_argument(
        "contact",
        action="store_true",
        help="get the attention alignment with the contact map",
    )
    on_set.add_argument(
        "instability",
        action="store_true",
        help="get the attention alignment with the instability index",
    )
    on_set.add_argument(
        "kmeans",
        action="store_true",
        help="get the attention alignment with the clusters found by the ",
        "k-means algorithm",
    )
    on_set.add_argument(
        "louvain",
        action="store_true",
        help="get the attention alignment with the communities found by the ",
        "louvain_partitions algorithm",
    )
    # optional arguments
    on_set.add_argument(
        "-s", "--save_every",
        default="none",  # if the flag is not present
        const="both",  # if the flag is present but no arguments are given
        nargs="?",
        choices=("none", "plot", "csv", "both"),
        help="save plots and/or csv files relative to each single peptide "
        "chain; if no arguments are given, both plots and csv files are saved",
    )
    on_set.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="verbose output: (-v) print info about the chain composition and "
        "the performed steps (-vv) for debugging",
    )

    # on_chain parser
    on_chain = subparsers.add_parser(
        "on_chain",
        help="get the attention alignment with one of the positional "
        "arguments for one peptide chain",
    )
    # positional arguments
    on_chain.add_argument(
        "chain_code",
        type=str,
        help="code of the input peptide chain",
    )
    # net_viz parser
    net_viz = subparser.add_parser(
        "net_viz",
        help="specify the network to visualize the analisys on the chain",
    )
    # positional
    net_viz.add_argument(
        "plot_type",
        type=str,
        choices=["chain3D", "pca", "network"],
        help="type of plot to visualize",
    )
    """# optional
    visualize.add_argument(
        "-r", "--print_results",
        action="store_true",
        help='decide if you want to visualize also the results of the analysis, '
        'both for the attention alignment, PCAs, and V-measure',
    )
    """
    # positional
    net_viz.add_argument(
        'analyse',
        type=str,
        choices=['louvain', 'kmeans', 'both', 'only_pca'],
        help='type of analysis to visualize the results on',
    )
    # increase option
    net_viz.add_argument(
        '-nc', '--node_color',
        type=str,
        default='ph_local',
        choices=['ph_local', 'ph_single', 'charge', 'flexy', 'iso_ph'],
        help='color of the node in the plots',
    )
    net_viz.add_argument(
        '-ec', '--edge_color',
        type=str,
        default="instability",
        choices=["instability", "proximity", "sequence_adjancency"],
        help='color of the edge in the plots',
    )
    net_viz.add_argument(
        "-es", "--edge_style",
        type=str,
        default="sequence_adjancency",
        choices=["instability", "proximity", "sequence_adjancency"],
    )
    net_viz.add_argument(
        '-n', '--node_size',
        type=str,
        default="volume",
        choices=["charge", "radius_of_gyration", "surface", "volume"],
        help='size of the node in the plots',
    )
    # optional arguments
    on_chain.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="verbose output: (-v) print info about the chain composition and "
        "the performed steps (-vv) for debugging",
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


def main():
    """Run the script chosen by the user."""
    args = parse_args()
    
    # this is the only logger and I do love cheesecakes
    log = Logger(name="cheesecake", verbosity=args.verbose)

    config = config_parser.Config("config.txt")
    paths = config.get_paths()

    file_folder = paths["FILE_FOLDER"]
    plot_folder = paths["PLOT_FOLDER"]
    
    file_dir = Path(__file__).resolve().parents[1]/file_folder
    plot_dir = Path(__file__).resolve().parents[1]/plot_folder

    file_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model_name = "Rostlab/prot_bert"
    with Loading("Loading the model"):
        model, tokenizer = load_model(model_name)

    if args.subparser == "on_set":
        proteins = config.get_proteins()
        protein_codes = proteins["PROTEIN_CODES"].split(" ")

        if protein_codes[0] == '':  
        # i.e., if PROTEIN_CODES is not provided in the configuration file
            max_length = proteins["MAX_LENGTH"]
            sample_size = proteins["SAMPLE_SIZE"]
            protein_codes = fetch_pdb_entries(
                max_length=max_length, n_results=sample_size
            )
            protein_codes_file = file_dir/"protein_codes.txt"
            with open(protein_codes_file, "w") as f:
                f.write(" ".join(protein_codes))
                log.logger.info(f"Protein codes saved to {protein_codes_file}")

        with Timer("Total running time"):
            for code_idx, code in enumerate(protein_codes):
                with (
                    Timer(f"Running time for [yellow]{code}[/yellow]"),
                    torch.no_grad(),
                ):
                    log.logger.info(f"Protein n.{code_idx+1}: [yellow]{code}")
                    attention, att_head_sum, CA_Atoms, amino_acid_df, \
                        att_to_aa = preprocess.main(
                            code, model, tokenizer, args.save_every
                        )

                    if len(CA_Atoms) <= 1:
                        log.logger.info(
                            f"Chain {code} has less than two valid residues..."
                            " Skipping"
                        )
                        # delete the code from protein_codes.txt
                        with open(protein_codes_file, "r") as file:
                            filedata = file.read()
                        filedata = filedata.replace(code+" ", "")
                        with open(protein_codes_file, "w") as file:
                            file.write(filedata)
                        continue

                    number_of_heads, number_of_layers = get_model_structure(
                        attention
                    )
                    chain_amino_acids = amino_acid_df["Amino Acid"].to_list()

                    head_att_align, layer_att_align = align_with_contact.main(
                        attention, CA_Atoms, chain_amino_acids, att_to_aa,
                        code, args.save_every
                    )

                    # instantiate the data structures to store the sum of the
                    # quantities to average over the set of proteins later
                    if code_idx == 0:
                        sum_amino_acid_df = pd.DataFrame(
                            data=0, index=all_amino_acids,
                            columns=["Amino Acid", "Total Occurrences"]
                        )
                        sum_amino_acid_df["Amino Acid"] = all_amino_acids
                        sum_att_head_sum = torch.zeros((
                            number_of_layers,
                            number_of_heads,
                        ))
                        sum_att_to_aa = torch.zeros((
                            len(all_amino_acids),
                            number_of_layers,
                            number_of_heads,
                        ))
                        sum_head_att_align = np.zeros((
                            number_of_layers,
                            number_of_heads,
                        ))
                        sum_layer_att_align = np.zeros((
                            number_of_layers,
                        ))

                    # sum all the quantities
                    # in order to sum the data frames, we merge them...
                    sum_amino_acid_df = pd.merge(
                        sum_amino_acid_df,
                        amino_acid_df[amino_acid_df.columns[:-2]],
                        on="Amino Acid", how='left'
                    )
                    # ... then we sum the columns...
                    sum_amino_acid_df[
                        "Total Occurrences"
                    ] = sum_amino_acid_df["Occurrences"].add(
                        sum_amino_acid_df["Total Occurrences"], fill_value=0
                    )
                    # ... and we drop the columns we don't need anymore
                    sum_amino_acid_df.drop(
                        columns=["Occurrences"], inplace=True
                    )

                    sum_att_head_sum = torch.add(
                        sum_att_head_sum, att_head_sum
                    )
                    sum_att_to_aa = torch.add(
                        sum_att_to_aa, att_to_aa
                    )
                    sum_head_att_align = np.add(
                        sum_head_att_align, head_att_align
                    )
                    sum_layer_att_align = np.add(
                        sum_layer_att_align, layer_att_align
                    )

            # rename the columns to the original shorter names
            sum_amino_acid_df.rename(
                columns={"Total Occurrences": "Occurrences"}, inplace=True
            )
            sum_amino_acid_df["Percentage Frequency (%)"] = (
                sum_amino_acid_df["Occurrences"]/
                sum_amino_acid_df["Occurrences"].sum()*100
            )
            sum_amino_acid_df["Total Occurrences"] = ""
            sum_amino_acid_df.at[0, "Total Occurrences"] = (
                sum_amino_acid_df["Occurrences"].sum()
            )
            log.logger.info(
                f"[bold white]GLOBAL DATA FRAME[/]\n{sum_amino_acid_df}"
            )
            sum_amino_acid_df.to_csv(
                file_dir/"total_residue_df.csv", index=False, sep=';'
            )

            """ sum_amino_acid_df and att_to_aa are built by considering the
            twenty possible amino acids, but some of them may not be present.
            Therefore, we drop the rows relative to the amino acids with zero
            occurrences, and the tensors relative to those amino acids
            """
            zero_indices = [
                idx for idx in range(len(sum_amino_acid_df)) if (
                    sum_amino_acid_df.at[idx, "Occurrences"] == 0
                )
            ]
            nonzero_indices = [
                idx for idx in range(len(sum_amino_acid_df)) if (
                    sum_amino_acid_df.at[idx, "Occurrences"] != 0
                )
            ]

            sum_amino_acid_df.drop(zero_indices, axis=0, inplace=True)
            sum_att_to_aa = torch.index_select(
                sum_att_to_aa, 0, torch.tensor(nonzero_indices)
            )

            glob_att_to_aa, glob_att_sim_df, avg_att_align = \
                compute_on_set.main(
                    sum_att_head_sum,
                    sum_att_to_aa,
                    sum_head_att_align,
                    sum_layer_att_align,
                    sum_amino_acid_df,
                )

            plotting.plot_on_set(
                glob_att_to_aa,
                glob_att_sim_df,
                avg_att_align,
                sum_amino_acid_df,
            )

    if args.subparser == "on_chain":
        seq_dir = plot_dir/args.code
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        with (
            Timer(f"Running time for [yellow]{args.code}[/yellow]"),
            torch.no_grad(),
        ):
          # TODO
          """if args.subparser == 'net_work':
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
            """
            attention, att_head_sum, CA_Atoms, amino_acid_df, att_to_aa = \
                preprocess.main(args.code, model, tokenizer, args.save_every)

            if len(CA_Atoms) <= 1:
                raise Exception(
                    "Chain {args.code} has less than two valid residues..."
                    " Aborting"
                )

            chain_amino_acids = amino_acid_df["Amino Acid"].to_list()

            head_att_align, layer_att_align = align_with_contact.main(
                attention, CA_Atoms, chain_amino_acids, att_to_aa, args.code,
                args.save_every
            )


if __name__ == '__main__':
    main()
