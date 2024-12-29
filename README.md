# Text mining 2024-25 Final Project: NER in the CSIRO Adverse Drug Event Corpus (CADEC)

## Installation

### Clone the Repository: 
```
git clone https://github.com/Rajivrocks-Ltd/text-mining-final.git
cd text-mining-final
```

The project has been implemented on two systems that both have CUDA (NVIDIA GPU) devices to run on. Some code cells might not run on a system without a cuda GPU, but these cells are not required to run the experiments (mostly memory fragmenting settings)


### Install the requirements: 

It is advised to recreate our environment using conda with the following command and the included yml file. This has the highest chance of working out of the box. 

```
conda env create -f environment.yml
```

A requirements.txt is also provided for users who do not use conda.

```
pip install -r requirements.txt
```

### Hardware background
All experiments were run on a workstation with the following OS and hardware:
- OS: Ubuntu 24.04
- CPU: Ryzen 9 7950x 16 core - 32 thread CPU
- GPU: RTX 3080
- RAM: 64GB DDR5 5600Mhz
- SSD: Samsung 980 Pro

- Python 3.12.8 (see yml for exact virtual environment used, cuda might not work out of the box with older GPU's, but the code will run (on CPU))
