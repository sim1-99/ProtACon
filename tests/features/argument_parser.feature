Feature: Argument Parser
    Parse the arguments passed from command line.

Scenario: Input 'on_set'
    Given the argument "on_set"
    When I parse the arguments
    Then "save_every" is set to "'none'"
    And "verbose" is set to "0"

Scenario: Input 'on_set --save_every -v'
    Given the argument "on_set"
    And the argument "--save_every"
    And the argument "-v"
    When I parse the arguments
    Then "save_every" is set to "'both'"
    And "verbose" is set to "1"

Scenario: Input 'on_set --save_every both -vv'
    Given the argument "on_set"
    And the argument "--save_every both"
    And the argument "-vv"
    When I parse the arguments
    Then "save_every" is set to "'both'"
    And "verbose" is set to "2"

Scenario: Input 'on_set --save_every csv'
    Given the argument "on_set"
    And the argument "--save_every csv"
    When I parse the arguments
    Then "save_every" is set to "'csv'"

Scenario: Input 'on_set --save_every plot'
    Given the argument "on_set"
    And the argument "--save_every plot"
    When I parse the arguments
    Then "save_every" is set to "'plot'"

Scenario: Input 'on_set --save_every none'
    Given the argument "on_set"
    And the argument "--save_every none"
    When I parse the arguments
    Then "save_every" is set to "'none'"

Scenario: Input 'on_chain 1HPV'
    Given the argument "on_chain 1HPV"
    When I parse the arguments
    Then "code" is set to "'1HPV'"
    And "verbose" is set to "0"

Scenario: Input 'on_chain 1HPV -v'
    Given the argument "on_chain 1HPV"
    And the argument "-v"
    When I parse the arguments
    Then "verbose" is set to "1"

Scenario: Input 'on_chain 1HPV -vv'
    Given the argument "on_chain 1HPV"
    And the argument "-vv"
    When I parse the arguments
    Then "verbose" is set to "2"