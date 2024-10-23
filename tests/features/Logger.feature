@Logger
Feature: Logger
    Log messages to stdout.

    Scenario: Get an existing Logger instance
        Given an instance of Logger
        When I call an existing Logger instance
        Then the instance I get is the expected one

    Scenario Outline: Log messages with a verbosity level
        Given an instance of Logger with level <verbosity> of verbosity
        When I log a warning message
        And I log an info message
        And I log a debug message
        Then the expected message <message> is printed

        Examples:
            | verbosity | message            |
            | 0         | warning            |
            | 1         | warning info       |
            | 2         | warning info debug |
