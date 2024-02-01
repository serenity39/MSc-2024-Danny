#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --job-name=bert_reranking
#SBATCH --output=bert_reranking.out
 
# Activate environment
uenv miniconda-python310
conda activate pygaggle

# Run the Python script
python -u ../code/bert_reranking.py
