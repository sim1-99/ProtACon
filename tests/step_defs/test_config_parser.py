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

from ProtACon.config_parser import Config


features_path = Path(__file__).resolve().parents[1]/"features"
test_data_path = Path(__file__).resolve().parents[1]/"test_data"
scenarios(str(features_path/"config_parser.feature"))

# Background steps
@given(
    "the path to the configuration file",
    target_fixture="config_file_path",
)
def config_file_path():
    return test_data_path/"config_test.txt"

# Given steps
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

# When steps
@when(
    "I create an instance of Config",
    target_fixture="Config_instance",
)
def Config_instance(config_file_path):
    return Config(config_file_path)

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

# Then steps
@then("the type of the instance is ConfigParser")
def Config_is_ConfigParser(Config_instance):
    assert isinstance(Config_instance, Config)

@then("the method returns a dictionary")
def return_dict(Config_method):
    assert isinstance(Config_method, dict)

@then(parsers.parse("the method returns the expected value {value}"))
def return_expected_value(value, Config_method, config_section, config_option):
    assert str(Config_method[config_section][config_option]) == value
