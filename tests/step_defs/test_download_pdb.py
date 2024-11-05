"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-05

Test suite for download_pdb.feature.

"""
from pathlib import Path

from Bio.PDB.PDBList import PDBList
from pytest_bdd import (
    given,
    scenarios,
    then,
    when,
)
import pytest

from ProtACon.modules.basics import download_pdb


features_path = Path(__file__).resolve().parents[1]/"features"
test_data_path = Path(__file__).resolve().parents[1]/"test_data"
scenarios(str(features_path/"download_pdb.feature"))

# Background steps
@given(
    "the path to the folder with the PDB files",
    target_fixture="pdb_files_path",
)
def pdb_files_path():
    return test_data_path

# Given steps
@given("a PDB code")
def pdb_code():
    return "1HPV"

@given("a list of PDB codes")
def pdb_codes():
    return ["1HPV", "4REF"]

# When steps
@when("I download the corresponding PDB file")
def download_pdb_file(pdb_files_path):
    download_pdb(pdb_code(), pdb_files_path)

@when(
    "I download the corresponding PDB files",
    target_fixture="out",
)
def download_pdb_files(capsys, pdb_files_path):
    """I need to capture the output of the function to check if the PDB FTP
    service is available.
    """
    PDBList().download_pdb_files(
        pdb_codes=pdb_codes(), file_format="pdb", pdir=pdb_files_path
    )
    return capsys.readouterr().out

@then("the file is saved in the folder with the PDB files")
def pdb_is_saved(pdb_files_path):
    assert (pdb_files_path/f"pdb{pdb_code().lower()}.ent").is_file()

@then("the files are saved in the folder with the PDB files")
def pdb_are_saved(out, pdb_files_path):
    if "Desired structure doesn't exists" in out:
        pytest.skip("The PDB FTP service is not available")
    for code in pdb_codes():
        assert (pdb_files_path/f"pdb{code.lower()}.ent").is_file()
