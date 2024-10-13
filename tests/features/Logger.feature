Feature: Logger
    Log information to stdout.

    Background:
        Given an instance of Logger

    Scenario: Get an existing Logger instance
        When I call an existing Logger instance
        Then the instance I get is the expected one

    Scenario: Log a message
        Given a message
        When I log the message
        Then the message is printed to stdout
