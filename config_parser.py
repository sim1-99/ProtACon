#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration parser."""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

import configparser
from os import makedirs


class Config:
    """A class of objects that can read data from a configuration file."""

    def __init__(
            self,
            filename: str
            ):
        """
        Contructor of the class.

        Parameters
        ----------
        filename : str
            name of the configuration file with the values

        Returns
        -------
        None.

        """
        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        self.config.read(filename)

    def get_cutoffs(
            self
            ) -> dict[str, float | int]:
        """
        Return a dictionary with the cutoffs for binarizing the contact map.

        Returns
        -------
        dict[str, float | int]
            dictionary that stores a str identifier and the cutoffs for the
            corresponding thresholding

        """
        return {
            "DISTANCE_CUTOFF": float(
                self.config.get("cutoffs", "DISTANCE_CUTOFF")),
            "POSITION_CUTOFF": int(
                self.config.get("cutoffs", "POSITION_CUTOFF"))
            }

    def get_paths(
            self
            ) -> dict[str, str]:
        """
        Return a dictionary with the paths to folders to store files.

        Returns
        -------
        dict[str, str]
            dictionary that stores a str identifier and the paths to the
            corresponding folder

        """
        return {"PDB_FOLDER": self.config.get("paths", "PDB_FOLDER"),
                "PLOT_FOLDER": self.config.get("paths", "PLOT_FOLDER")
                }

    def get_proteins(
            self
            ) -> dict[str, str]:
        """
        Return a dictionary with the codes representing the peptide chains.

        Returns
        -------
        dict[str, str]
            dictionary that stores a str identifier and a tuple with the
            protein codes
        """
        return {"PROTEIN_CODES": self.config.get("proteins", "PROTEIN_CODES")}


def ensure_storage_directories_exist(
        paths: dict[str, str]
        ) -> None:
    """
    Ensure that the target directories to store files exist.

    It either creates them if they are absent or leaves the target directories
    unchanged if already present.

    Parameters
    ----------
    paths : dict[str, str]
        dictionary with the paths to folders to store files

    Returns
    -------
    None.

    """
    makedirs(paths["PDB_FOLDER"], exist_ok=True)
    makedirs(paths["PLOT_FOLDER"], exist_ok=True)
