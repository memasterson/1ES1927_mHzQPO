#!/bin/bash

# example sbatch script to run the QPO simulations for significance testing
# this example is for the Feb. 2023 data (ObsID 0915390701) with the power-law broadband noise model
# it will spawn 10 jobs across different nodes, each of which will be parallelized across 16 cores
# qpo_sims.py has hard coded the location of the necessary broadband h5 MCMC files
# the data path needs to point to where the 20s light curve data files sit

#SBATCH --ntasks-per-node=16 # number of cores per node
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-08:00 # maximum run time
#SBATCH --mem-per-cpu=500 # 500 MB of memory per CPU
#SBATCH --array=1-10 # spawns 10 jobs (will be split among 10 nodes, each with 16 cores)

source activate qpo # activate conda environment

echo "CPUs allocated on this node: $SLURM_CPUS_ON_NODE"

# run the code, but run 100,000 / 10 (i.e. n_sim_total / n_array) simulations since this will be run for EACH array element

# Feb. 2023, 2-10 keV
python qpo_sims.py --obsid 0915390701 --data_path ../data/lightcurves/ --emin 2.0 --emax 10.0 --tbin 20 --n 0 --n_sim 10000 --n_jobs $SLURM_CPUS_ON_NODE --psd_model powerlaw --maxlike_init False
