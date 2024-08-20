# 1ES1927_mHzQPO
Data, code, and figures for the paper on the mHz QPO in 1ES 1927+654

The data (including light curves from XMM-Newton and NICER and the XMM-Newton spectra plotted in Figure 1) and  can be found in the data/ folder. The code used for analysis and figure creation is in the code/ folder.

Contents of code/:
- timing.py: python script containing various functions commonly used in the other files (including PSD creation + binning and fitting functions for the PSDs), this code uses the pylag python package to produce PSDs and runs on standard light curve outputs of XMM-Newton EPIC-pn data
- Masterson24b_paper_figures.ipynb: python notebook that produces the figures in the paper, all cells run on timescales of 10s of seconds, except the WD accretion stream calculation cell (Supp. Fig. 7) which takes of order 5 minutes
- broadband.ipynb: python notebook that fits the 2-10 keV power spectra with broadband noise models, corner plots and overplotted models on the PSDs are shown inline, stores the resulting MCMC h5 files in the code/broadband_mcmcs/ folder, when these chains already exist they will be loaded in when the code is run (not re-computed), with the MCMCs already run and h5 files ready to load this code takes of order minutes to run (if they need to be run MCMCs take on the order of 10 minutes per MCMC with the current computing set up)
- qpo_lorfit.ipynb: python notebook that fits the 2-10 keV power spectra (and some 0.2-3 power spectra) with an additional lorentzian component on top of the broadband noise model to measure the properties of the QPO (e.g. centroid frequency, RMS, Q), corner plots and overplotted models on the PSDs are shown inline, stores the resulting MCMC h5 files in the code/qpo_mcmcs/ folder along with the resulting QPO frequency, RMS, and Q factors in numpy arrays, run times are similar to broadband.ipynb
- qpo_sims.py: python script to run Timmer & Koenig simulations of XMM light curves to assess the QPO significance, this script is designed for parallelization and running on a cluster, an example sbatch shell script to run the code is given in example_run_qpo_sims.sh and takes on the order of a few hours when run with the given cluster set up, the code outputs resulting numpy arrays with various statistics used to assess the QPO significance
- combine_TK.py: python script that combines the output of the qpo_sims.py run into a single output array that is saved in the TK_results/ folder, this is to consolidate the many (10) jobs that are run with qpo_sims.py spawning across different nodes, run time is on the order of seconds

Software & system requirements:
All of the code is python-based and has been run on MacOS with python 3.9.18. The following python packages are required, with their tested versions listed next to them.
astropy    5.3.4
corner     2.2.1
emcee      3.1.2
h5py       3.6.0
joblib     1.2.0
matplotlib 3.8.0
numpy      1.26.3
pandas     1.4.4
pylag      2.2
scipy      1.13.1
