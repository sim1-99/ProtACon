#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils.

This module contains:
    - the implementation of a timer
    - a function for averaging together pandas dataframes or numpy arrays in a
      list
    - a function for normalizing numpy arrays
    - a function for reading the .pdb files
"""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from contextlib import contextmanager
from datetime import datetime
from functools import reduce
import logging
from pathlib import Path
from rich.console import Console

from ProtACon import config_parser

from Bio.PDB.Structure import Structure
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pandas as pd


@contextmanager
def Loading(
        message: str
        ) -> None:
    """
    Implement loading animation.

    Parameters
    ----------
    message : str
        text to print during the animation

    Returns
    -------
    None.

    """
    console = Console()
    try:
        with console.status(f"[bold green]{message}..."):
            yield
    finally:
        console.log(f"[bold green]{message}... Done")


@contextmanager
def Timer(
        description: str
        ) -> None:
    """
    Implement timer.

    Parameters
    ----------
    description : str
        text to print

    Returns
    -------
    None.

    """
    start = datetime.now()
    try:
        yield
    finally:
        end = datetime.now()
        timedelta = end-start
        message = (f"{description}, started: {start}, ended: {end}, elapsed:"
                   f"{timedelta}")
        logging.warning(message)


def average_maps_together(
        list_of_maps: list[pd.DataFrame | np.ndarray]
        ) -> pd.DataFrame | np.ndarray:
    """
    Average together the maps (tensors or arrays) contained in a list.

    Parameters
    ----------
    list_of_maps : list[pd.DataFrame | np.ndarray]
        contains the maps to be averaged together

    Returns
    -------
    average_map : pd.DataFrame | np.ndarray

    """
    if type(list_of_maps[0]) is pd.DataFrame:
        average_map = reduce(lambda x, y: x.add(y, fill_value=0), list_of_maps)
        average_map.div(len(list_of_maps))
        """  # TODO: remove if not use
        average_map = torch.sum(
            torch.stack(list_of_maps), dim=0)/len(list_of_maps)
        """
        return average_map

    if type(list_of_maps[0]) is np.ndarray:
        average_map = np.sum(np.stack(list_of_maps), axis=0)/len(list_of_maps)
        return average_map


def normalize_array(
        array: np.ndarray
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
        seq_ID: str
        ) -> Structure:
    """
    Download the .pdb file of the sequence ID to get its structure.

    Parameters
    ----------
    seq_ID : str
        alphanumerical code representing uniquely one peptide chain

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        object containing information about each atom of the peptide chain

    """
    config = config_parser.Config("config.txt")
    paths = config.get_paths()
    pdb_folder = paths["PDB_FOLDER"]
    pdb_dir = Path(__file__).parent.parent.parent/pdb_folder

    pdb_import = PDBList()
    pdb_file = pdb_import.retrieve_pdb_file(
        pdb_code=seq_ID, file_format="pdb", pdir=pdb_dir)

    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(seq_ID, pdb_file)

    return structure
