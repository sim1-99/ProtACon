"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-10-13

Test suite for Logger.feature.

"""
from pathlib import Path

from pytest_bdd import (
    given,
    scenarios,
    parsers,
    then,
    when,
)

from ProtACon.modules.utils import Logger

features_path = Path(__file__).resolve().parents[1]/"features"
scenarios(str(features_path/"Logger.feature"))

# Given steps
@given(
    "an instance of Logger",
    target_fixture="mylog",
)
def log():
    return Logger(name="mylog")

@given(
    parsers.parse("an instance of Logger with level {verbosity} of verbosity"),
    target_fixture="log_verb",
    converters={"verbosity": int},
)
def log_verb(verbosity):
    return Logger(name="mylog_verb", verbosity=verbosity)

# When steps
@when(
    "I call an existing Logger instance",
    target_fixture="log",
)
def get_logger(mylog):
    return mylog.get_logger()

@when("I log a warning message")
def log_W(log_verb):
    log_verb.logger.propagate = True  # otherwise caplog.txt will be empty
    log_verb.logger.warning("warning")

@when("I log an info message")
def log_I(log_verb):
    log_verb.logger.info("info")

@when("I log a debug message")
def log_D(log_verb):
    log_verb.logger.debug("debug")

# Then steps
@then("the instance I get is the expected one")
def check_logger_instance(mylog, log):
    assert log is mylog

@then(parsers.parse("the expected message {message} is printed"))
def print_expected_message(message, caplog):
    assert message == " ".join(caplog.messages)
