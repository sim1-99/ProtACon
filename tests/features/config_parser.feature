Feature: Configuration Parser
    Parse the configuration variables in config.txt.

    Scenario: Create a ConfigParser object
        Given the path to the configuration file
        When I create an instance of Config
        Then the type of the instance is ConfigParser

    Scenario Outline: Get the configuration variables
        Given an instance of Config
        And the section <section>
        And the option <option>
        When I call the corresponding method
        Then it returns a dictionary
        And it returns the expected value <value>

        Examples: Table
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

