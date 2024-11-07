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
    return ["1VXA", "4REF"]


# When steps
@when("I download the corresponding PDB file")
def download_pdb_file(pdb_files_path):
    download_pdb(pdb_code(), pdb_files_path)
    yield
    # Teardown
    Path.unlink(pdb_files_path/f"pdb{pdb_code().lower()}.ent")


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
    out = capsys.readouterr().out
    yield out
    # Teardowm
    if "Desired structure doesn't exists" not in out:
        for code in pdb_codes():
            Path.unlink(pdb_files_path/f"pdb{code.lower()}.ent")


@then("the file is saved in the folder with the PDB files")
def pdb_is_saved(pdb_files_path):
    assert (pdb_files_path/f"pdb{pdb_code().lower()}.ent").is_file()


@then("the files are saved in the folder with the PDB files")
def pdb_are_saved(out, pdb_files_path):
    if "Desired structure doesn't exists" in out:
        pytest.skip("The PDB FTP service is not available")
    for code in pdb_codes():
        assert (pdb_files_path/f"pdb{code.lower()}.ent").is_file()


@then("the file saved is the expected one")
def pdb_is_expected(pdb_files_path):
    with open(pdb_files_path/f"pdb{pdb_code().lower()}.ent") as f:
        assert f.readline().startswith("HEADER")
        f.seek(0)  # go back to the first line
        assert pdb_code() in f.readline()


@then("the files saved are the expected ones")
def pdb_are_expected(out, pdb_files_path):
    if "Desired structure doesn't exists" in out:
        pytest.skip("The PDB FTP service is not available")
    for code in pdb_codes():
        with open(pdb_files_path/f"pdb{code.lower()}.ent") as f:
            assert f.readline().startswith("HEADER")
            f.seek(0)  # go back to the first line
            assert code in f.readline()
