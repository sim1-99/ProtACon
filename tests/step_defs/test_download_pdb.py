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
    parsers,
    scenarios,
    then,
    when,
)
import pytest

from ProtACon.modules.basics import download_pdb


scenarios("download_pdb.feature")


# Background steps
@given(
    "the path to the folder with the PDB files",
    target_fixture="pdb_files_path",
)
def pdb_files_path(data_path):
    return data_path


@given(
    parsers.parse("a PDB ID {chain_ID}"),
    target_fixture="chain_ID",
)
def pdb_code(chain_ID):
    return chain_ID


@given(
    parsers.parse("a list of PDB IDs {chain_IDs}"),
    target_fixture="chain_IDs",
)
def pdb_codes(chain_IDs):
    return chain_IDs.split(" ")


# When steps
@when("I download the corresponding PDB file")
def download_pdb_file(chain_ID, pdb_files_path):
    download_pdb(chain_ID, pdb_files_path)
    yield
    # Teardown
    Path.unlink(pdb_files_path/f"pdb{chain_ID.lower()}.ent")


@when(
    "I download the corresponding PDB files",
    target_fixture="out",
)
def download_pdb_files(capsys, chain_IDs, pdb_files_path):
    """I need to capture the output of the function to check if the PDB FTP
    service is available.
    """
    PDBList().download_pdb_files(
        pdb_codes=chain_IDs, file_format="pdb", pdir=pdb_files_path
    )
    out = capsys.readouterr().out
    yield out
    # Teardowm
    if "Desired structure doesn't exists" not in out:
        for code in chain_IDs:
            Path.unlink(pdb_files_path/f"pdb{code.lower()}.ent")


@then("the file is saved in the folder with the PDB files")
def pdb_is_saved(chain_ID, pdb_files_path):
    assert (pdb_files_path/f"pdb{chain_ID.lower()}.ent").is_file()


@then("the files are saved in the folder with the PDB files")
def pdb_are_saved(chain_IDs, out, pdb_files_path):
    if "Desired structure doesn't exists" in out:
        pytest.skip("The PDB FTP service is not available")
    for code in chain_IDs:
        assert (pdb_files_path/f"pdb{code.lower()}.ent").is_file()


@then("the file saved is the expected one")
def pdb_is_expected(chain_ID, pdb_files_path):
    with open(pdb_files_path/f"pdb{chain_ID.lower()}.ent") as f:
        assert f.readline().startswith("HEADER")
        f.seek(0)  # go back to the first line
        assert chain_ID in f.readline()


@then("the files saved are the expected ones")
def pdb_are_expected(chain_IDs, out, pdb_files_path):
    if "Desired structure doesn't exists" in out:
        pytest.skip("The PDB FTP service is not available")
    for code in chain_IDs:
        with open(pdb_files_path/f"pdb{code.lower()}.ent") as f:
            assert f.readline().startswith("HEADER")
            f.seek(0)  # go back to the first line
            assert code in f.readline()
