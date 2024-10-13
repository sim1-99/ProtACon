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
    then,
    when,
)


features_path = Path(__file__).resolve().parents[1]/"features"
scenarios(str(features_path/"Logger.feature"))

# Given steps
@given(
    "a message",
    target_fixture="message",
)
def message():
    return "This is a message"

# When steps
@when(
    "I call an existing Logger instance",
    target_fixture="mylog",
)
def get_logger(log):
    return log.get_logger()

@when("I log the message")
def log_message(log, message):
    log.logger.propagate = True  # otherwise caplog.txt will be empty
    log.logger.warning(message)

# Then steps
@then("the instance I get is the expected one")
def check_logger_instance(mylog, log):
    assert mylog is log

@then("the message is printed to stdout")
def print_log_message(caplog, message):
    assert message in caplog.text
