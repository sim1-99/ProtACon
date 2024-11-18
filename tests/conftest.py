"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-07

Collection of common pytest fixtures.

"""
from pathlib import Path
import configparser
import os
import warnings

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.PDBParser import PDBParser
import pytest

from ProtACon.config_parser import Config
from ProtACon.modules.basics import download_pdb


class TestingConfig(Config):
    """Testing version of the Config class."""

    def __init__(self):
        self.config = configparser.ConfigParser()

        self.config["cutoffs"] = {
            "ATTENTION_CUTOFF": "0.1",
            "DISTANCE_CUTOFF": "8.0",
            "POSITION_CUTOFF": "6",
        }
        self.config["paths"] = {
            "PDB_FOLDER": "pdb_files",
            "FILE_FOLDER": "files",
            "PLOT_FOLDER": "plots",
        }
        self.config["proteins"] = {
            "PROTEIN_CODES": "1HPV 1AO6",
            "MIN_LENGTH": "15",
            "MAX_LENGTH": "300",
            "MIN_RESIDUES": "10",
            "SAMPLE_SIZE": "1000",
        }


@pytest.fixture(scope="session")
def TestingConfigInstance():
    """Instance of the TestingConfig class."""
    return TestingConfig()


@pytest.fixture(scope="session")
def data_path():
    """Path to the directory containing the PDB files."""
    return Path(__file__).resolve().parent/"test_data"


@pytest.fixture(scope="session")
def chain_ID():
    """The PDB ID of a peptide chain."""
    return "2ONX"


@pytest.fixture(scope="session")
def structure(chain_ID, data_path):
    """Structure of a peptide chain."""
    download_pdb(chain_ID, data_path)
    pdb_path = data_path/f"pdb{chain_ID.lower()}.ent"

    with warnings.catch_warnings():
        # warn that the chain is discontinuous, this is not a problem though
        warnings.simplefilter('ignore', PDBConstructionWarning)
        structure = PDBParser().get_structure(chain_ID, pdb_path)

    yield structure
    # Teardown
    test_files = os.listdir(data_path)
    for item in test_files:
        if item.endswith(".ent"):
            os.remove(os.path.join(data_path, item))


@pytest.fixture(scope="session")
def model_name():
    """Name of the ProtBert model."""
    return "Rostlab/prot_bert"
