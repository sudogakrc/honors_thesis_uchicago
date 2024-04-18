#!/bin/bash

#SBATCH --job-name=model
#SBATCH --output=models/outputs/left_right/slurm-%A_%a_model.out
#SBATCH --error=model.err
#SBATCH --partition=general
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --array=0

conda create --name neural-sentiment python=3.6.8

export PATH="/home/sudogakaraca/miniconda/bin:$PATH"

pip install -r /net/projects/ycleong/neuralsentiment/requirements/requirements.txt

conda activate neural-sentiment

python test.py $SLURM_ARRAY_TASK_ID
