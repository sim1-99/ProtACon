@config_parser
Feature: Configuration Parser
    Parse the configuration variables.

    Scenario: Create a ConfigParser object
        When I create an instance of Config
        Then the type of the instance is ConfigParser

    Scenario Outline: Get the configuration variables
        Given the section <section>
        And the option <option>
        When I create an instance of Config
        And I call the corresponding method
        Then the method returns a dictionary
        And the method returns the expected value <value>

        Examples:
            | section  | option           | value     |
            | cutoffs  | ATTENTION_CUTOFF | 0.1       |
            | cutoffs  | DISTANCE_CUTOFF  | 8.0       |
            | cutoffs  | POSITION_CUTOFF  | 6         |
            | paths    | PDB_FOLDER       | pdb_files |
            | paths    | FILE_FOLDER      | files     |
            | paths    | PLOT_FOLDER      | plots     |
            | proteins | PROTEIN_CODES    | 1HPV 1AO6 |
            | proteins | MIN_LENGTH       | 15        |
            | proteins | MAX_LENGTH       | 300       |
            | proteins | MIN_RESIDUES     | 10        |
            | proteins | SAMPLE_SIZE      | 1000      |
