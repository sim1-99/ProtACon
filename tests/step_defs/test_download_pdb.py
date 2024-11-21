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


# Given steps
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
def download_pdb_file(
    chain_ID, mocker, mock_rcsb_response, pdb_files_path, resp_headers
):
    """
    -> mock_rcsb_response is passed to mock the response of requests.get() in
    download_pdb().
    -> mocker.mock_open() acts in download_pdb().
    -> resp_headers["Content-Disposition"] is equal to
    f"attachment; filename={chain_ID}.pdb". I am simulating the reading of that
    string just like in a real PDB file.

    """
    mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data=resp_headers["Content-Disposition"]),
    )
    download_pdb(chain_ID, pdb_files_path)


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


# Then steps
@then("the expected URL is called")
def pdb_url_is_called(url, response_url):
    assert response_url == url


@then("the URL is called once")
def pdb_url_is_called_once(response_call_count):
    assert response_call_count == 1


@then("the call to the URL is successful")
def pdb_call_is_successfull(response_status_code):
    assert response_status_code == 200


@then("the file downloaded is the expected one")
def pdb_is_expected(chain_ID, pdb_files_path):
    # open() is mocked with mocker.mock_open(), the file does not really exist
    with open(pdb_files_path/f"pdb{chain_ID.lower()}.ent") as f:
        assert chain_ID.casefold() in f.readline()


@then("the files are saved in the folder with the PDB files")
def pdb_are_saved(chain_IDs, out, pdb_files_path):
    if "Desired structure doesn't exists" in out:
        pytest.skip("The PDB FTP service is not available")
    for code in chain_IDs:
        assert (pdb_files_path/f"pdb{code.lower()}.ent").is_file()


@then("the files saved are the expected ones")
def pdb_are_expected(chain_IDs, out, pdb_files_path):
    if "Desired structure doesn't exists" in out:
        pytest.skip("The PDB FTP service is not available")
    for code in chain_IDs:
        with open(pdb_files_path/f"pdb{code.lower()}.ent") as f:
            assert f.readline().startswith("HEADER")
            f.seek(0)  # go back to the first line
            assert code in f.readline()
