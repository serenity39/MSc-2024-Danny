#!/bin/bash

#SBATCH --job-name=hello_world
#SBATCH --output=hello_world.out
#SBATCH --time=5:00
#SBATCH --ntasks=1
#SBATCH --mem=100MB

module load python39
python3 hello_world.py
