"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-09-28

Useful test steps common to many scenarios.

"""
from pytest_bdd import given
from pathlib import Path

from ProtACon.modules.utils import Logger


test_data_path = Path(__file__).resolve().parents[1]/"test_data"

# Shared Given Steps
@given(
    "the path to the configuration file",
    target_fixture="config_file_path",
)
def config_file_path():
    return test_data_path/"config_test.txt"

@given(
    "an instance of Logger",
    target_fixture="log",
)
def log():
    return Logger(name="mylog")
