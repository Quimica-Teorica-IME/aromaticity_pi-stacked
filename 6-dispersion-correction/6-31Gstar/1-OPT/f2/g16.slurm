#!/bin/bash
##
## Copyright (C) 2009-2021 VersatusHPC, Inc.
##
#SBATCH --job-name=f2_pi-stacked
##
## partition = fila
#SBATCH --partition=normal
##
## nodes = quantidade de nodes
#SBATCH --nodes=1
##
## ntasks-per-node = quantidade de nucleos por node
#SBATCH --ntasks-per-node=32
##
## time = quantidade de tempo
#SBATCH --time=720:00:00
##

g16root="/sw/cluster/pub/apps/"
g16dir="/sw/cluster/pub/apps/g16"
LaunchDir=$PWD;
LocalRunDir=/data/$USER/$SLURM_JOB_ID
export GAUSS_SCRDIR=$LocalRunDir;
mkdir -p $GAUSS_SCRDIR
export GAUSS_EXEDIR=$g16dir
export GAUSS_ARCHDIR=$g16dir
source $g16root/g16/bsd/g16.profile
export PGI_FASTMATH_CPU=sandybridge

g16 < f2.gjf  > f2.log

