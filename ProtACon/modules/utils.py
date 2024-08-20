"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

This module contains:
    - the implementation of a timer
    - a function for averaging together pandas dataframes or numpy arrays in a
      list
    - a function for normalizing numpy arrays
    - a function for reading the .pdb files
    
"""
from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Iterator
import logging

from Bio.PDB.Structure import Structure
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from rich.console import Console
import numpy as np
import pandas as pd
import torch

from ProtACon import config_parser


@contextmanager
def Loading(
    message: str,
) -> Iterator[None]:
    """
    Implement loading animation.

    Parameters
    ----------
    message : str
        The text to print during the animation.

    Returns
    -------
    None

    """
    console = Console()
    try:
        with console.status(f"[bold green]{message}..."):
            yield
    finally:
        console.log(f"[bold green]{message}... Done")


@contextmanager
def Timer(
    description: str,
) -> Iterator[None]:
    """
    Implement timer.

    Parameters
    ----------
    description : str
        The text to print.

    Returns
    -------
    None

    """
    start = datetime.now()
    try:
        yield
    finally:
        end = datetime.now()
        timedelta = end-start
        message = (
            f"{description}, started: {start}, ended: {end}, elapsed: "
            f"{timedelta}"
        )
        logging.warning(message)


def average_arrs_together(
    list_of_arrs: list[np.ndarray],
) -> np.ndarray:
    """
    Average together the numpy arrays contained in a list.

    Parameters
    ----------
    list_of_arrs : list[np.ndarray]
        The arrays to average together.

    Returns
    -------
    average_arr : np.ndarray

    """
    average_arr = np.sum(np.stack(list_of_arrs), axis=0)/len(list_of_arrs)

    return average_arr


def average_dfs_together(
    list_of_dfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Average together the dataframes contained in a list.

    Parameters
    ----------
    list_of_dfs : list[pd.DataFrame]
        The dataframes to average together.

    Returns
    -------
    average_df : pd.DataFrame

    """
    average_df = reduce(lambda x, y: x.add(y, fill_value=0), list_of_dfs)
    average_df.div(len(list_of_dfs))

    return average_df


def normalize_array(
    array: np.ndarray,
) -> np.ndarray:
    """
    Normalize a numpy array.

    Parameters
    ----------
    array : np.ndarray

    Returns
    -------
    norm_array : np.ndarray

    """
    if True in np.isnan(array):
        array_max, array_min = np.nanmax(array), np.nanmin(array)
    else:
        array_max, array_min = np.max(array), np.min(array)
    norm_array = (array - array_min)/(array_max - array_min)

    return norm_array


def read_pdb_file(
    seq_ID: str,
) -> Structure:
    """
    Download the .pdb file of the sequence ID to get its structure.

    Parameters
    ----------
    seq_ID : str
        The alphanumerical code representing uniquely the peptide chain.

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        The object containing information about each atom of the peptide chain.

    """
    config = config_parser.Config("config.txt")
    paths = config.get_paths()
    pdb_folder = paths["PDB_FOLDER"]
    pdb_dir = Path(__file__).resolve().parents[2]/pdb_folder

    pdb_import = PDBList()
    pdb_file = pdb_import.retrieve_pdb_file(
        pdb_code=seq_ID, file_format="pdb", pdir=pdb_dir
    )

    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(seq_ID, pdb_file)

    return structure
