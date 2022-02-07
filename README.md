# KaggleFold

### Making Protein folding accessible to all via Kaggle!

| Notebooks | monomers | complexes | mmseqs2 | jackhmmer | templates   |
| :-------- | -------  | --------- | ------- | --------- | ----------- |
| [AlphaFold2_mmseqs2](https://www.kaggle.com/victorfernandezalbor/alphafold2-kagglefold/) | Yes | Yes | Yes | No | Yes | 
| [AlphaFold2_batch](https://www.kaggle.com/victorfernandezalbor/alphafold2-kagglefold/) | Yes | Yes | Yes | No | Yes | 
| [RoseTTAFold](https://www.kaggle.com/victorfernandezalbor/alphafold2-kagglefold/) | Yes | No | Yes | No | No | 
| [AlphaFold2](https://www.kaggle.com/victorfernandezalbor/alphafold2-kagglefold/) (from Deepmind) | Yes | Yes | No | Yes | No | 
||
| [AlphaFold2_advanced](https://www.kaggle.com/victorfernandezalbor/alphafold2-kagglefold/) | Yes | Yes | Yes | Yes | No |
||


### FAQ
- Can I use the models for **Molecular Replacement**?
  - Yes, but be **CAREFUL**, the bfactor column is populated with pLDDT confidence values (higher = better). Phenix.phaser expects a "real" bfactor, where (lower = better). See [post](https://twitter.com/cheshireminima/status/1423929241675120643) from Claudia Millán.
- What is the maximum length?
  - For GPU: `Tesla T4` or `Tesla P100` with ~16G the max length is ~1400
  - For GPU: `Tesla K80` with ~12G the max length is ~1000
  - To check what GPU you got, open a new code cell and type `!nvidia-smi`
- Is it okay to use the MMseqs2 MSA server (`cf.run_mmseqs2`) on a local computer?
  - You can access the server from a local computer if you queries are serial from a single IP. Please do not use multiple computers to query the server.
- Where can I download the databases used by KaggleFold?
  - The databases are available at [kagglefold.mmseqs.com](https://kagglefold.mmseqs.com)
- I want to render my own images of the predicted structures, how do I color by pLDDT?
  - In pymol for AlphaFold structures: `spectrum b, red_yellow_green_cyan_blue, minimum=50, maximum=90`
  - In pymol for RoseTTAFold structures: `spectrum b, red_yellow_green_cyan_blue, minimum=0.5, maximum=0.9`
- What is the difference between the AlphaFold2_advanced and AlphaFold2_mmseqs2 (_batch) notebook for complex prediction? 
  - We currently have two different ways to predict protein complexes: (1) using the AlphaFold2 model with residue index jump and (2) using the AlphaFold2-multimer model. AlphaFold2_advanced supports (1) and AlphaFold2_mmseqs2 (_batch) (2).
- What is the difference between localkagglefold and the pip installable kagglefold_batch?
  -  localkagglefold is a command line interface for our advanced notebooks. pip is a command line version of the alphafold_mmseqs2 and alphafold_batch notebook.


### Running locally

_Note: If you need amber or templates, checkout [localkagglefold](https://github.com/YoshitakaMo/localkagglefold) instead_

Install KaggleFold using the `pip` commands below. `pip` will resolvei and install all required dependencies and KaggleFold should be ready within a few minutes to use. Please check the [JAX documentation](https://github.com/google/jax#pip-installation-gpu-cuda) for how to get JAX to work on your GPU or TPU.

```shell
pip install "kagglefold[alphafold] @ git+https://github.com/vfalbor/KaggleFold"
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
```

```shell
kagglefold_batch <directory_with_fasta_files> <result_dir> 
```

If no GPU or TPU is present, `kagglefold_batch` can be executed (slowly) using only a CPU with the `--cpu` parameter.

### Generating MSAs

First create a directory for the databases on a disk with sufficient storage (940GB (!)). Depending on where you are, this will take a couple of hours: 

```shell
./setup_databases.sh /path/to/db_folder
```

Download and unpack mmseqs (Note: The required features aren't in a release yet, so currently, you need to compile the latest version from source yourself). If mmseqs is not in your `PATH`, replace `mmseqs` below with the path to your mmseqs:

```shell
# This needs a lot of CPU
kagglefold_search input_sequences.fasta /path/to/db_folder search_results
# This just does a bit of IO
kagglefold_split_msas search_results msas
# This needs a GPU
kagglefold_batch msas predictions
```

This will create intermediate folders `search_results` and `msas` that you can eventually delete, and a `predictions` folder with all pdb files. 


### Acknowledgments
- We would like to thank the [RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold) and [AlphaFold](https://github.com/deepmind/alphafold) team for doing an excellent job open sourcing the software. 
- Also credit to [David Koes](https://github.com/dkoes) for his awesome [py3Dmol](https://3dmol.csb.pitt.edu/) plugin, without whom these notebooks would be quite boring!
- A colab by Sergey Ovchinnikov (@sokrypton), Milot Mirdita (@milot_mirdita) and Martin Steinegger (@thesteinegger).

### CITE some of the authors of these work

- Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S and Steinegger M. KaggleFold - Making protein folding accessible to all. <br />
  bioRxiv (2021) doi: [10.1101/2021.08.15.456425](https://www.biorxiv.org/content/10.1101/2021.08.15.456425v2)
- If you’re using **AlphaFold**, please also cite: <br />
  Jumper et al. "Highly accurate protein structure prediction with AlphaFold." <br />
  Nature (2021) doi: [10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- If you’re using **AlphaFold-multimer**, please also cite: <br />
  Evans et al. "Protein complex prediction with AlphaFold-Multimer." <br />
  biorxiv (2021) doi: [10.1101/2021.10.04.463034v1](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)
- If you are using **RoseTTAFold**, please also cite: <br />
  Minkyung et al. "Accurate prediction of protein structures and interactions using a three-track neural network." <br />
  Science (2021) doi: [10.1126/science.abj8754](https://doi.org/10.1126/science.abj8754)

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.5123296.svg)](https://doi.org/10.5281/zenodo.5123296)

-----------------

