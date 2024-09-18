"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

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
from ProtACon import average_on_set
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
        help="get attention alignment and other quantities averaged over a set"
        " of peptide chains",
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
        help="get attention alignment and other quantities for one single "
        "peptide chain",
    )
    # positional arguments
    on_chain.add_argument(
        "code",
        type=str,
        help="code of the input protein",
    )
    # optional arguments
    on_chain.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="verbose output: (-v) print info about the chain composition and "
        "the performed steps (-vv) for debugging",
    )

    # 3d_viz parser
    net_viz = subparsers.add_parser(
        "net_viz",
        help="visualize 3D network of a protein with one selected property "
        "and the attention alignment of that property",
    )
    # positional arguments
    net_viz.add_argument(
        "code",
        type=str,
        help="code of the input protein",
    )
    net_viz.add_argument(
        "property",
        type=str,
        help="property or network to show",
    )
    # optional arguments
    net_viz.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="verbose output: (-v) print info about the chain composition and "
        "the performed steps (-vv) for debugging",
    )

    args = parser.parse_args()

    return args


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
                with Timer(
                    f"Running time for [yellow]{code}[/yellow]"
                ), torch.no_grad():

                    log.logger.info(f"Protein n.{code_idx+1}: [yellow]{code}")
                    attention, att_head_sum, CA_Atoms, amino_acid_df, \
                        att_to_aa = preprocess.main(
                            code, model, tokenizer, args.save_every
                        )

                    number_of_heads, number_of_layers = get_model_structure(
                        attention
                    )
                    chain_amino_acids = amino_acid_df["Amino Acid"].to_list()

                    # instantiate the data structures to store the sum of the
                    # quantities to average over the set of proteins later
                    if code_idx == 0:
                        sum_amino_acid_df = pd.DataFrame(
                            data=0, index=all_amino_acids,
                            columns=[
                                "Amino Acid",
                                "Total Occurrences",
                            ]
                        )
                        sum_amino_acid_df["Amino Acid"] = all_amino_acids
                        sum_att_head_sum = torch.zeros(
                            (
                                number_of_layers,
                                number_of_heads
                            ), dtype=float
                        )
                        sum_att_to_aa = torch.zeros(
                            (
                                len(all_amino_acids),
                                number_of_layers,
                                number_of_heads
                            ), dtype=float
                        )
                        sum_head_att_align = np.zeros(
                            (number_of_layers, number_of_heads), dtype=float
                        )
                        sum_layer_att_align = np.zeros(
                            number_of_layers, dtype=float
                        )

                    head_att_align, layer_att_align = align_with_contact.main(
                        attention, CA_Atoms, chain_amino_acids, att_to_aa,
                        code, args.save_every
                    )

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
                        sum_layer_att_align, layer_att_align)

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

            PT_att_to_aa, PWT_att_to_aa, PH_att_to_aa, glob_att_sim_df, \
                avg_head_att_align, avg_layer_att_align = average_on_set.main(
                    sum_att_head_sum,
                    sum_att_to_aa,
                    sum_head_att_align,
                    sum_layer_att_align,
                    sum_amino_acid_df,
                )

            plotting.plot_on_set(
                PT_att_to_aa,
                PWT_att_to_aa,
                PH_att_to_aa,
                glob_att_sim_df,
                avg_head_att_align,
                avg_layer_att_align,
                sum_amino_acid_df,
            )

    if (args.subparser == "on_chain" or args.subparser == "net_viz"):
        seq_dir = plot_dir/args.code
        seq_dir.mkdir(parents=True, exist_ok=True)

    if args.subparser == "on_chain":
        with Timer(
            f"Running time for [yellow]{args.code}[/yellow]"
        ), torch.no_grad():

            attention, att_head_sum, CA_Atoms, amino_acid_df, att_to_aa = \
                preprocess.main(
                    args.code, model, tokenizer, args.save_every
                )

            chain_amino_acids = amino_acid_df["Amino Acid"].to_list()

            head_att_align, layer_att_align = align_with_contact.main(
                attention, CA_Atoms, chain_amino_acids, att_to_aa, args.code,
                args.save_every
            )


if __name__ == '__main__':
    main()
