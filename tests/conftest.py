"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-07

Collection of pytest fixtures.

"""
from pathlib import Path
import warnings

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.PDBParser import PDBParser
import pytest

from ProtACon.modules.basics import download_pdb


@pytest.fixture(scope="session")
def data_path():
    """Path to the directory containing the PDB files."""
    return Path(__file__).resolve().parent/"test_data"


@pytest.fixture(scope="session", params=["1HPV", "2UX2", "4REF", "9RSA"])
def chain_ID(request):
    """The PDB ID of a peptide chain."""
    return request.param


@pytest.fixture(scope="module")
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
    Path.unlink(data_path/f"pdb{chain_ID.lower()}.ent")
