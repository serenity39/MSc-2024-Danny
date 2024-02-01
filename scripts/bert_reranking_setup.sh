#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --job-name=bert_reranking_setup
#SBATCH --output=bert_reranking_setup.out

# Set up environment
uenv miniconda-python39
conda create -n pygaggle pip python=3.10 -y

# Activate environment
conda activate pygaggle

# Install necassary packages
pip install --upgrade pip
python3 -m pip install --upgrade setuptools
pip install tensorflow
conda install -c conda forge pytorch faiss-cpu -y
pip install faiss-gpu
cd ..
pip install -r requirements.txt
cd ../pygaggle
pip install -r requirements.txt
pip install -e .
