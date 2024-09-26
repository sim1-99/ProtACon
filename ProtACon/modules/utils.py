"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

This module contains:
    - the definition of the class Logger
    - the implementation of a timer
    - the implementation of a loading animation
    - a function for normalizing numpy arrays
    - a function for reading the .pdb files
    - a funtion for changing the default format of the warnings
    
"""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator
import logging

from Bio.PDB.Structure import Structure
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from rich.console import Console
from rich.logging import RichHandler
import numpy as np

from ProtACon import config_parser


class Logger:
    """
    A class of objects that can log information to a file with the desired
    verbosity.

    """
    def __init__(
        self,
        name: str,
        verbosity: int = 0,
    ):
        """
        Contructor of the class.

        Parameters
        ----------
        name : str
            The name to call the logger with.
        verbosity : int = 0
            The level of verbosity. 0 set the logging level to WARNING, 1 to
            INFO and 2 to DEBUG.

        """
        self.name = name
        self.verbosity = verbosity

        loglevel = 30 - 10*verbosity

        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.logger.setLevel(loglevel)

        self.formatter = logging.Formatter(
            fmt='%(message)s',
            datefmt='[%H:%M:%S]',
        )

        self.handler = RichHandler(markup=True, rich_tracebacks=True)
        self.handler.setFormatter(self.formatter)

        if self.logger.handlers:
            self.logger.handlers.clear()

        self.logger.addHandler(self.handler)

    def get_logger(
        self,
    ):
        """
        Get from the Logger object with a given name, the attributes previously
        used to set the corresponding logger, in order to get the same logger.

        """
        handler = self.handler
        handler.setFormatter(self.formatter)

        self.logger = logging.getLogger(self.name)

        return self


log = Logger("mylog").get_logger()


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
            f"{description}, [green]started[/green]: {start},"
            f" [red]ended[/red]: {end}, [cyan]elapsed[/cyan]: {timedelta}"
        )
        log.logger.info(message)


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

# UNUSED FUNCTIONS:
# average_arrs_together, average_dfs_together, warning_on_one_line

'''from functools import reduce

import pandas as pd


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


def warning_on_one_line(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    """
    Change the default format of the warnings.

    Parameters
    ----------
    message : Warning | str
        The message to print.
    category : type[Warning]
        The type of warning.
    filename : str
        The name of the file where the warning is raised.
    lineno : int
        The line number in the file where the warning is raised.
    line : str | None = None

    Returns
    -------
    str
        The formatted warning message.

    """
    return '%s: %s\n' % (category.__name__, message)
'''
