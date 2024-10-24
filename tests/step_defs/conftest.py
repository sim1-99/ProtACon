from pytest_bdd import given
from pathlib import Path


test_data_path = Path(__file__).resolve().parents[1]/"test_data"

# Shared Given Steps
@given(
    "the path to the configuration file",
    target_fixture="config_file_path",
)
def config_file_path():
    return test_data_path/"config_test.txt"
