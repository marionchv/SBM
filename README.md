# SBM: Stochastic Boltzmann Machine
Tested with Python 3.7.3

<!-- ## Scientific publications
If you intend to publish a paper that utilizes any portion of this code, please don't hesitate to get in touch with me. I would be delighted to engage in a discussion regarding your work and its connection to this project. Your contributions and insights are greatly valued. -->

## Prerequisites

### Create a virtual env

First of all, I recommend creating a virtual environment. Here's an example using virtualenv:

```
pip3 install virtualenv
virtualenv -p python3 env_SBM
source env_SBM/bin/activate
```

### Install requirements

Install python and python library from requirements.txt: 
```
pip3 install -r requirements.txt --no-cache-dir
```

### Compile C_MonteCarlo module:

```
sh src/SBM/MonteCarlo/MCMC_Potts/make_mcmc_Potts.sh
```

### Install the editable version of SBM :

```
pip3 install -e .
```

## Dataset format

Before using SBM to infer fields and couplings from a MSA you need to load your fasta file and turn this fasta file into a numpy array of size (Number of sequences x Protein length)

```
import SBM.utils.utils as ut
MSA = ut.load_fasta('fasta_file')
np.save('data/MSA_array/MSA_fam.npy',MSA)
```

## Training

See demo_SBM for an example

## Example inside data folder

```
data/
├── fasta
├── MSA_array
	└── MSA_CM.npy
├── Ind_train
	└── Ind_train_CM.npy
```


## use jupyter notebook on a cluster

```
pip3 install ipykernel
````

Then you can create a notebook and choose python environment env_SBM