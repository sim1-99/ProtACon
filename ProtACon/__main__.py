"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

__main__.py file for command line application.

"""
import argparse
from pathlib import Path

from Bio.PDB.PDBList import PDBList
import torch

from ProtACon import config_parser
from ProtACon.modules.basics import (
    download_pdb,
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
from ProtACon import manage_tot_ds
from ProtACon import plotting
from ProtACon import preprocess


def parse_args(args: list[str] = None):
    """
    Argument parser.
    
    Parameters
    ----------
    args: list
        arguments to parse, only used when testing

    """
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

    args = parser.parse_args(args)

    return args


def main():
    """Run the script chosen by the user."""
    args = parse_args()

    log = Logger(name="mylog", verbosity=args.verbose)

    config_file_path = Path(__file__).resolve().parents[1]/"config.txt"
    config = config_parser.Config(config_file_path)

    paths = config.get_paths()
    pdb_folder = paths["PDB_FOLDER"]
    file_folder = paths["FILE_FOLDER"]
    plot_folder = paths["PLOT_FOLDER"]

    pdb_dir = Path(__file__).resolve().parents[1]/pdb_folder
    file_dir = Path(__file__).resolve().parents[1]/file_folder
    plot_dir = Path(__file__).resolve().parents[1]/plot_folder

    pdb_dir.mkdir(parents=True, exist_ok=True)
    file_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    proteins = config.get_proteins()
    min_residues = proteins["MIN_RESIDUES"]

    model_name = "Rostlab/prot_bert"
    with Loading("Loading the model"):
        model, tokenizer = load_model(model_name)

    if args.subparser == "on_set":
        protein_codes = proteins["PROTEIN_CODES"].split(" ")

        if protein_codes[0] == '':  
        # i.e., if PROTEIN_CODES is not provided in the configuration file
            min_length = proteins["MIN_LENGTH"]
            max_length = proteins["MAX_LENGTH"]
            sample_size = proteins["SAMPLE_SIZE"]
            protein_codes = fetch_pdb_entries(
                min_length=min_length,
                max_length=max_length,
                n_results=sample_size,
                stricter_search=False,
            )

        protein_codes_file = file_dir/"protein_codes.txt"
        with open(protein_codes_file, "w") as f:
            f.write(" ".join(protein_codes))
            log.logger.info(f"Protein codes saved to {protein_codes_file}")

        PDBList().download_pdb_files(
            pdb_codes=protein_codes, file_format="pdb", pdir=pdb_dir
        )

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
                    log.logger.info(
                        f"Actual number of residues: {len(CA_Atoms)}"
                    )

                    skips = 0
                    if len(CA_Atoms) < min_residues:
                        log.logger.warning(
                            f"Chain {code} has less than {min_residues} valid "
                            "residues... Skipping"
                        )
                        skips += 1
                        # delete the code from protein_codes.txt
                        with open(protein_codes_file, "r") as file:
                            filedata = file.read()
                        filedata = filedata.replace(code+" ", "")
                        with open(protein_codes_file, "w") as file:
                            file.write(filedata)
                        continue

                    chain_amino_acids = amino_acid_df["Amino Acid"].to_list()

                    head_att_align, layer_att_align = align_with_contact.main(
                        attention, CA_Atoms, chain_amino_acids, att_to_aa,
                        code, args.save_every
                    )

                    chain_ds = (
                        amino_acid_df,
                        att_head_sum,
                        att_to_aa,
                        head_att_align,
                        layer_att_align,
                    )

                    # instantiate the data structures to store the sum of the
                    # quantities to average over the set of proteins later
                    if code_idx == 0:
                        n_heads, n_layers = get_model_structure(attention)
                        tot_amino_acid_df, tot_att_head_sum, tot_att_to_aa, \
                            tot_head_att_align, tot_layer_att_align = \
                                manage_tot_ds.create(n_layers, n_heads)

                    # sum all the quantities
                    tot_amino_acid_df, tot_att_head_sum, tot_att_to_aa, \
                        tot_head_att_align, tot_layer_att_align = \
                            manage_tot_ds.update(
                                tot_amino_acid_df,
                                tot_att_head_sum,
                                tot_att_to_aa,
                                tot_head_att_align,
                                tot_layer_att_align,
                                chain_ds,
                            )

            tot_amino_acid_df = manage_tot_ds.append_frequency_and_total(
                tot_amino_acid_df
            )

            log.logger.info(
                f"[bold white]GLOBAL DATA FRAME[/]\n{tot_amino_acid_df}"
            )
            tot_amino_acid_df.to_csv(
                file_dir/"total_residue_df.csv", index=False, sep=';'
            )
            """ tot_amino_acid_df and tot_att_to_aa are built by considering
            twenty possible amino acids, but some of them may not be present.
            Therefore, we drop the rows relative to the amino acids with zero
            occurrences, and the tensors relative to those amino acids.
            """
            tot_amino_acid_df, tot_att_to_aa = manage_tot_ds.keep_nonzero(
                tot_amino_acid_df, tot_att_to_aa
            )

            sample_size = len(protein_codes) - skips
            glob_att_to_aa, glob_att_sim, avg_att_align = compute_on_set.main(
                tot_amino_acid_df,
                tot_att_head_sum,
                tot_att_to_aa,
                tot_head_att_align,
                tot_layer_att_align,
                sample_size,
                )

            plotting.plot_on_set(
                tot_amino_acid_df,
                glob_att_to_aa,
                glob_att_sim,
                avg_att_align,
            )

    if args.subparser == "on_chain":
        seq_dir = plot_dir/args.code
        seq_dir.mkdir(parents=True, exist_ok=True)

        download_pdb(args.code, pdb_dir)

        with (
            Timer(f"Running time for [yellow]{args.code}[/yellow]"),
            torch.no_grad(),
        ):
            attention, att_head_sum, CA_Atoms, amino_acid_df, att_to_aa = \
                preprocess.main(args.code, model, tokenizer, save_opt="both")

            if len(CA_Atoms) < min_residues:
                raise Exception(
                    f"Chain {args.code} has less than {min_residues} valid "
                    "residues... Aborting"
                )

            chain_amino_acids = amino_acid_df["Amino Acid"].to_list()

            head_att_align, layer_att_align = align_with_contact.main(
                attention, CA_Atoms, chain_amino_acids, att_to_aa, args.code,
                save_opt="both"
            )


if __name__ == '__main__':
    main()
