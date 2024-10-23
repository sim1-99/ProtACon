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
@given(parsers.parse('the command "{command}"'))
def append_command(command, args):
    args.append(command)

@given(parsers.parse('the flag "{flag}"'))
def append_flag(flag, args):
    args.append(flag)

@given(parsers.parse('the value "{value}"'),)
def append_value(value, args):
    # add the value of the flag to the same sting
    args[-1] = ' '.join([args[-1], value])
    return args

# When steps
@when(
    "I parse the arguments",
    target_fixture="out",
)
def parse_arguments(args, capsys):
    print(parse_args(args))
    return capsys.readouterr().out

# Then steps
@then(parsers.parse('the flag "{flag}" is set to "{value}"'))
def check_args(out, flag, value):
    message = f"{flag[2:]}={value}"
    assert message in out
