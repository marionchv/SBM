# SBM: Stochastic Boltzmann Machine
Tested with Python 3.7.3

<!-- ## Scientific publications
If you intend to publish a paper that utilizes any portion of this code, please don't hesitate to get in touch with me. I would be delighted to engage in a discussion regarding your work and its connection to this project. Your contributions and insights are greatly valued. -->

## Prerequisites

Install python and python library from requirements.txt: 
```
pip3 install -r requirements.txt --no-cache-dir
```

compile C_MonteCarlo module:
```
sh src/SBM/MonteCarlo/MCMC_Potts/make_mcmc_Potts.sh
```

# To install the editable version of SBM use :
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

See demo_SBM

## Example inside data folder

```
data/
├── fasta
├── MSA_array
	└── MSA_CM.npy
├── Ind_train
	└── Ind_train_CM.npy
```


## Create a virtual env

```
pip3 install virtualenv
virtualenv -p python3 env_name
source env_name/bin/activate
```

## use jupyter notebook on a cluster


```
pip3 install ipykernel
````
create a notebook
choose python environment env_name