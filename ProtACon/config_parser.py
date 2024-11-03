"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

Configuration parser.

"""
import configparser
from pathlib import Path


class Config:
    """A class of objects that can read data from a configuration file."""

    def __init__(
        self,
        file_path: Path,
    ) -> None:
        """
        Contructor of the class.

        Parameters
        ----------
        file_path : Path
            The path to the configuration file.

        Returns
        -------
        None

        """
        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        self.config.read(file_path)

    def get_cutoffs(
        self,
    ) -> dict[str, float | int]:
        """
        Return a dictionary with the cutoffs for thresholding the attention
        matrices and for binarizing the contact map.

        Returns
        -------
        dict[str, float | int]
            The identifier and the cutoffs for the corresponding thresholdings.

        """
        return {
            "ATTENTION_CUTOFF": float(
                self.config.get("cutoffs", "ATTENTION_CUTOFF")
            ),
            "DISTANCE_CUTOFF": float(
                self.config.get("cutoffs", "DISTANCE_CUTOFF")
            ),
            "POSITION_CUTOFF": int(
                self.config.get("cutoffs", "POSITION_CUTOFF")
            ),
            "INSTABILITY_CUTOFF": float(
                self.config.get("cutoffs", "INSTABILITY_CUTOFF")
            ),
            "STABILITY_CUTOFF": float(
                self.config.get("cutoffs", "STABILITY_CUTOFF")
            ),
        }

    def get_paths(
        self,
    ) -> dict[str, str]:
        """
        Return a dictionary with the paths to folders to store the files.

        Returns
        -------
        dict[str, str]
            The identifier and the paths to the corresponding folder.

        """
        return {
            "PDB_FOLDER": self.config.get("paths", "PDB_FOLDER"),
            "FILE_FOLDER": self.config.get("paths", "FILE_FOLDER"),
            "PLOT_FOLDER": self.config.get("paths", "PLOT_FOLDER"),
            "NET_FOLDER": self.config.get("paths", "NET_FOLDER"),
            "TEST_FOLDER": self.config.get("paths", "TEST_FOLDER"),
        }

    def get_proteins(
        self,
    ) -> dict[str, str | int]:
        """
        Return a dictionary with either the codes representing the proteins to
        run the experiment, or the parameters to fetch the codes from PDB.

        Returns
        -------
        dict[str, str | int]
            The identifier and the list of protein codes, the min and the max
            length that a protein can have, the minimum number of actual
            residues and the protein sample size.

        """
        return {
            "PROTEIN_CODES": self.config.get("proteins", "PROTEIN_CODES"),
            "MIN_LENGTH": int(self.config.get("proteins", "MIN_LENGTH")),
            "MAX_LENGTH": int(self.config.get("proteins", "MAX_LENGTH")),
            "MIN_RESIDUES": int(self.config.get("proteins", "MIN_RESIDUES")),
            "SAMPLE_SIZE": int(self.config.get("proteins", "SAMPLE_SIZE")),
        }
