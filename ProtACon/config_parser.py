#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration parser."""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

import configparser
from pathlib import Path


class Config:
    """A class of objects that can read data from a configuration file."""

    def __init__(
        self,
        filename: str
    ) -> None:
        """
        Contructor of the class.

        Parameters
        ----------
        filename : str
            name of the configuration file with the values

        Returns
        -------
        None

        """
        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        self.config.read(Path(__file__).resolve().parent/filename)

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
                self.config.get("cutoffs", "POSITION_CUTOFF")),
            "INSTABILITY_CUTOFF": float(
                self.config.get("cutoffs", "INSTABILITY_CUTOFF")),
            "STABILITY_CUTOFF": float(
                self.config.get("cutoffs", "STABILITY_CUTOFF"))
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
        return {
            "PDB_FOLDER": self.config.get("paths", "PDB_FOLDER"),
            "PLOT_FOLDER": self.config.get("paths", "PLOT_FOLDER"),
            "NET_FOLDER": self.config.get("networks", "NET_FOLDER")
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
