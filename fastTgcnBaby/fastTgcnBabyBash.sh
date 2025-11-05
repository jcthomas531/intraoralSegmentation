#!/bin/bash
#$ -l ngpus=1
#$ -pe smp 64
#$ -o fastTgcnBabyOut.o
#$ -e fastTgcnBabyError.e



apptainer exec ../../../containers/pytorch2.sif python train.py
