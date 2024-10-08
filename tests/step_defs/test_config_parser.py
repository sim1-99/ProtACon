"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-09-28

Test suite for config_parser.feature.

"""
from pathlib import Path
from pytest_bdd import (
    given, 
    parsers,
    scenarios,
    then,
    when,
)
import pytest

from ProtACon.config_parser import Config

features_path = Path(__file__).resolve().parents[1]/"features"
test_data_path = Path(__file__).resolve().parents[1]/"test_data"

scenarios(features_path/"config_parser.feature")

@pytest.fixture
def config_file_name():
    return "config_test.txt"

@pytest.fixture
def config_file_path(config_file_name):
    return test_data_path/config_file_name

# Scenario: Create a ConfigParser object
@given("the path to the configuration file")

@when(
    "I create an instance of Config",
    target_fixture="Config_instance",
)
def Config_instance(config_file_path):
    return Config(config_file_path)

@then("the type of the instance is ConfigParser")
def Config_is_ConfigParser(Config_instance):
    assert isinstance(Config_instance, Config)

# Scenario Outline: Get the configuration variables
@given(
    "an instance of Config",
    target_fixture="Config_instance",
)
def Config_instance(config_file_path):
    return Config(config_file_path)

@given(
    parsers.parse("the section {section}"),
    target_fixture="config_section",
)
def config_section(section):
    return section

@given(
    parsers.parse("the option {option}"),
    target_fixture="config_option",
)
def config_option(option):
    return option

@when(
    "I call the corresponding method",
    target_fixture="Config_method",
)
def Config_method(Config_instance):
    return {
        "cutoffs": Config_instance.get_cutoffs(),
        "paths": Config_instance.get_paths(),
        "proteins": Config_instance.get_proteins(),
    }

@then("it returns a dictionary")
def return_dict(Config_method):
    assert isinstance(Config_method, dict)

@then(parsers.parse("it returns the expected value {value}"))
def return_expected_value(
    Config_method, config_section, config_option, value
):
    assert str(Config_method[config_section][config_option]) == value
