# ProtACon
**ProtACon** is a project by Simone Chiarella and Renato Eliasy for the courses of Pattern Recognition and Complex Networks, MSc Physics, University of Bologna.

## Installation
To install ProtACon, clone the repository in some folder. In this folder, launch:

`pip install ProtACon`

Once you installed it, you can run the application from any path.

To install the repo in developer mode, run this command instead:

`pip install -e .`

This installation gives you the chance of editing the code without having to reinstall it every time to make changes effective.

If you also want to install the required packages to run the test suite, run:

`pip install .'[test]'`

or, in developer mode:

`pip install -e .'[test]'`

## Usage
You can launch different scripts by typing in the command line `ProtACon` followed by one of the following commands:

- `on_set` Compute the attention alignment with the contact between the residues of each peptide chain contained in a specified set. The attention alignment and the pairwise attention similarity of each peptide chain are finally averaged together.
- `on_chain` Do the same as `on_set`, but on a single protein, to specify using its unique identification code.

From the configuration file `config.txt`, it is possible to set the folder names where to store the plots and the PDB files of the proteins, and also the cutoffs for the thresholding of the contact maps. From there, you can also specify the set of proteins that you want to process with the command `on_set`.

## Other installation commands

