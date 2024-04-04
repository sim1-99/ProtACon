# ProtACon
**ProtACon** is a project by Simone Chiarella and Renato Eliasy for the courses of Pattern Recognition and Complex Networks, MSc Physics, University of Bologna.

To install it, clone the repository in some folder. In this folder, launch:

pip --editable ProtACon

Once you installed it, you can run different scripts by typing in the command line `ProtACon` followed by one of the following commands:

- `on_set` Compute the attention alignment with the contact between the residues of each peptide chain contained in a specified set. The attention alignment and the pairwise attention similarity of each peptide chain are finally averaged together.
- `on_chain` Do the same as `on_set`, but on a single protein, to specify using its unique identification code.
- `net_viz` Visualize a network showing the 3D structure of one protein, together with one specified property (pH, charge, contact), and the corresponding alignment with the attention given to each residue (still to implement).
