@download_pdb
Feature: Download PDB
    Download PDB files from RCSB Protein Data Bank, given one or multiple PDB
    codes.

    The second scenario is not a generalization of the first one, rather a test
    for the function PDBList().download_pdb_files. This function depends on the
    PDB FTP service, which is not always available. For this reason, in its
    test an exception handler is used to pass the test if the FTP service is
    not working. However, in the main code there is the possibility to download
    multiple PDB files using the function tested in the first scenario. In this
    case the function is just looped over the list of PDB codes.

    Background:
        Given the path to the folder with the PDB files

    Scenario: Download a PDB file
        Given a PDB code
        When I download the corresponding PDB file
        Then the file is saved in the folder with the PDB files
        And the file saved is the expected one

    Scenario: Download multiple PDB files
        Given a list of PDB codes
        When I download the corresponding PDB files
        Then the files are saved in the folder with the PDB files
        And the files saved are the expected ones
