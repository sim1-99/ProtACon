# ProtACon

**ProtACon** is a wide project aimed to explore and interpret the representations generated by transformers when they are applied to proteins. Take a look at our [experiment report]() to discover what you can do with ProtACon. 🧑‍🔬

> [!NOTE]
> You are on the main, stable branch. Check out the ["advanced"](https://github.com/sim1-99/ProtACon/tree/advanced) branch for an extended (yet less documented) version with more features.

The goal is to detect possible connections and similarities between the attention weights generated by the [ProtBert](https://huggingface.co/Rostlab/prot_bert) transformer, and the physico-chemical properties of the proteins that are fed into it.

This project was inspired by the work of Jesse Vig and colleagues ["BERTology Meets Biology"](https://arxiv.org/abs/2006.15222), which proved that BERT-based models are able to capture high-level structural properties of proteins, just by providing them with the sequence of amino acids in the protein chains.

👉 Check out the code documentation at the [reference guide](https://protacon.readthedocs.io/en/latest/).

## How ProtACon works

The whole pipeline is founded on two pillars: the ProtBert encoder for the extraction of the attention, and the [RCSB Protein Data Bank](https://www.rcsb.org) to get the protein structural information.

Starting from a PDB entry&mdash;a 4-digits alphanumerical code uniquely identifying one protein&mdash;the PDB file of the corresponding protein is downloaded. Proteins often have more than one chain, so only the first one is picked. From that the sequence of amino acids of the residues in the protein is stored, together with the coordinates of the &alpha;-carbon atom of each residue. The amino acid sequence is then passed to the ProtBert encoder, where it is processed and from which attention from each head of the model is extracted.

## What ProtACon does

ProtACon has two possible uses: either on single peptide chains, or on a set of them.

For both uses, the main results you get from the run are the **attention alignment** with the protein contact map and the **attention similarity** between amino acids&mdash;go look at [our report]() for their definitions. Beside that, a bunch of other quantities are computed and saved in dedicated folders.

The run on a set of chains does not save on your device the quantities relative to the single chains&mdash;unless the contrary is provided&mdash;but computes and stores the averages of those quantities relative to the whole protein set. Find the [guide to the output files](https://github.com/sim1-99/ProtACon/wiki/Guides#output-overview) in the wiki section.

ProtACon integrates the [PDB Search API](https://search.rcsb.org/#search-services) in its pipeline. Thus, when running on sets of proteins, you have two ways to choose the composition of the set:

- by passing the complete list of PDB entries;
- by providing some parameters for a search in the PDB API, such as the minimum and maximum numbers of residues in each chain, and the numbers of chains making up the set.

Look at the wiki section for more information about [configuring your experiment](https://github.com/sim1-99/ProtACon/wiki/Tutorials#configure-your-experiment).

<!-- markdownlint-disable -->
<p float="left">
  <img src="docs/pictures/avg_att_align_heads_6.png" width="300" />
  <img src="docs/pictures/att_sim.png" width="300" />
</p>

## Quickstart

### Prerequisites

Two prerequisites are needed:

- An environment with **Python-3.10.15** 🐍.
- The [GCC](https://gcc.gnu.org/) package installed&mdash;it is required for Biopython to work correctly.

  - If you are on a conda environment, you can install it with:

    ```bash
    conda install conda-forge::gcc
    ```

  - Otherwise, you can run the following command (requires root privileges):

    ```bash
    apt-get install gcc
    ```

  A correct functioning of the code was verified with `gcc-11.4.0` and `gcc-14.2.0`.

### Installation

To install ProtACon, execute the following commands:

```bash
git clone https://github.com/sim1-99/ProtACon.git`
cd ProtACon
```

Then, install with:

```bash
pip install .
```

Once you installed it, you can run the application from any path.

To install the repo in developer mode, run this command instead:

```bash
pip install -e .
```

This installation gives you the chance of editing the code without having to reinstall it every time to make changes effective.

If you also want to install the required packages to run the test suite, run:

```bash
pip install .'[test]'
```

or, in developer mode:

```bash
pip install -e .'[test]'
```

The [list of dependecies](https://github.com/sim1-99/ProtACon/blob/9ae08bef9e5a7d1f8591c9f886930d08d7f07d9a/pyproject.toml#L9C1-L21C2) downloaded can be found in pyproject.toml.

### Running the code

You can launch different scripts by typing in the command line `ProtACon` followed by one of the following commands.

- `on_chain`

  If you want to analyze a single protein, namely [6NJC](https://www.rcsb.org/structure/6NJC), then run:

  ```bash
  ProtACon on_chain 6NJC
  ```

  *Optional flags*

  `-v`, `--verbose`: print to the terminal more info about the run.
  ___

- `on_set`

  If you want to perform an analysis on a set of proteins, then run:

  ```bash
  ProtACon on_set
  ```

  *Optional flags*

  `-s`, `--save_every`: choose one between ["none", "plot", "csv", "both"] to selectively save files relative to the single peptide chains in the set. Files relative to the set are always saved instead.

  - If `--save_every` is not added at all, it is like "none" is passed, and no files relative to single chains are saved.
  - If `--save_every plot` is added, save the plots relative to every single chain in the set in a dedicated folder.
  - If `--save_every csv` is added, save the csv files with the amino acid occurrences in every single chain in the set in a dedicated folder.
  - If `--save_every` is added with no options, "both" is passed, which would be like passing both "plot" and "csv".

  `-v`, `--verbose`: print to the terminal more info about the run.
  ___

> [!WARNING]
> ProtACon does not overwrite existing plots. If you run the code passing the same plot folder as a previous run, no plots will be saved.

## Running tests

> [!IMPORTANT]
> Running tests requires you to include the section `'[test]'` when installing (see the [dedicated section](#installation)).

When in the main ProtACon folder, you can run all the tests just with the command `pytest`, as well as tests on single modules and functions by launching:

```bash
pytest -m <marker>
```

\<marker> being one of the [markers](https://github.com/sim1-99/ProtACon/blob/9ae08bef9e5a7d1f8591c9f886930d08d7f07d9a/pyproject.toml#L54C1-L87C2) in section \[tool.pytest.ini_options] of pyproject.toml.

Finally, running:

```bash
pytest .
```

will also run the plugin [pytest-pycodestyle](https://pypi.org/project/pytest-pycodestyle/), checking the code against [PEP8](https://peps.python.org/pep-0008/) style conventions.
