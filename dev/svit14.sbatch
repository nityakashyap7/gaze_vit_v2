#!/usr/bin/env bash
set -euo pipefail

#SBATCH --partition=nlp
#SBATCH --mem=1G
#SBATCH --output=/project2/biyik_1165/nityakas/slurm-logs/debug/slurm-%j-${MODEL}.out
#SBATCH --mail-user=nityakas@usc.edu  # <--- your email here
#SBATCH --mail-type=BEGIN,END,FAIL

