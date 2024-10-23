Feature: Argument Parser
    Parse the arguments passed from command line.

Scenario: Input 'on_set'
    Given the command "on_set"
    When I parse the arguments
    Then the flag "--save_every" is set to "'none'"
    And the flag "--verbose" is set to "0"

Scenario: Input 'on_set --save_every'
    Given the command "on_set"
    And the flag "--save_every"
    When I parse the arguments
    Then the flag "--save_every" is set to "'both'"

Scenario: Input 'on_set -v'
    Given the command "on_set"
    And the flag "-v"
    When I parse the arguments
    Then the flag "--verbose" is set to "1"

Scenario: Input 'on_set -vv'
    Given the command "on_set"
    And the flag "-vv"
    When I parse the arguments
    Then the flag "--verbose" is set to "2"
