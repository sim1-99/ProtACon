"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-10-19

Test suite for argument_parser.feature.

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

from ProtACon.__main__ import parse_args

features_path = Path(__file__).resolve().parents[1]/"features"
scenarios(str(features_path/"argument_parser.feature"))

# Fixtures
@pytest.fixture
def args():
    return []

# Given steps
@given(parsers.parse('the argument "{argument}"'))
def append_argument(argument, args):
    argument = argument.split()
    args.extend(argument)

# When steps
@when(
    "I parse the arguments",
    target_fixture="out",
)
def parse_arguments(args, capsys):
    print(parse_args(args))
    return capsys.readouterr().out

# Then steps
@then(parsers.parse('"{argument}" is set to "{value}"'))
def check_args(out, argument, value):
    message = f"{argument}={value}"
    assert message in out
