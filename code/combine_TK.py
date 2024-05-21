# load packages
import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.fft
import os
import astropy.stats as st
from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, NullFormatter
from matplotlib import ticker
# from redshift get d_L
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
# emcee stuff
import emcee
import corner
from scipy.stats import spearmanr, linregress
import pylag
from scipy.fft import ifft
from joblib import Parallel, delayed
from scipy.stats import binned_statistic, chi2
import argparse
import glob 

# set up plotting defaults
plt.rc('font', family='sans')
params = {
   'axes.labelsize': 45,
   'axes.linewidth': 3,
   'legend.fontsize': 30,
   'legend.frameon': True,
   'legend.fancybox': False,
   'legend.framealpha': 0.8,
   'legend.edgecolor': 'k',
   'lines.linewidth': 2,
   'font.size': 40,
   'font.weight': 'normal',
   'xtick.direction': 'in',
   'xtick.labelsize': 35,
   'xtick.color':'k',
   'xtick.major.bottom': True,
   'xtick.major.pad': 10,
   'xtick.major.size': 18,
   'xtick.major.width': 2,
   'xtick.minor.bottom': True,
   'xtick.minor.pad': 10,
   'xtick.minor.size': 9,
   'xtick.minor.top': True,
   'xtick.minor.visible': True,
   'xtick.minor.width': 2,
   'xtick.top': True,
   'ytick.direction': 'in',
   'ytick.labelsize': 35,
   'ytick.left': True,
   'ytick.right': True,
   'ytick.major.pad': 10,
   'ytick.major.size': 18,
   'ytick.major.width': 2,
   'ytick.minor.pad': 3.5,
   'ytick.minor.size': 9,
   'ytick.minor.visible': True,
   'ytick.minor.width': 2,
   'text.usetex': False,
   'figure.figsize': [10,10],
   'savefig.dpi': 500,
   }
plt.rcParams.update(params)

def combine_TK_results(obs, emin, emax, tbin, psd_model='broken_powerlaw'):

    # this should grab all of the files from different tasks
    def_str = obs+'_'+str(emin)+'-'+str(emax)+'keV_'+str(tbin)+'s_'+psd_model
    res_files = glob.glob('TK_results/'+def_str+'/TK_'+def_str+'*.npy')
    # SSE_files = glob.glob('TK_results/SSE_TK_'+obs+'_'+str(emin)+'-'+str(emax)+'keV_'+str(tbin)+'s_'+psd_model+'*.npy')

    combined_ndet_R = 0
    combined_ndet_SSE = 0
    combined_nsim = 0

    # SSE_comp = 862.763757922386
    
    # Load and append the results from each file
    for i,file in enumerate(res_files):
        data = np.load(file)
        combined_ndet_R = combined_ndet_R + data[1]
        combined_nsim = combined_nsim + data[-1]
        combined_ndet_SSE = combined_ndet_SSE + data[3]
        
        if i == 0:
            print('R from observation: ', data[0])
            print('SSE from observation: ', data[2])

        # print(i)
        # SSE_TK = np.load(SSE_files[i])
        # ndet_SSE = len(np.where(SSE_TK > SSE_comp)[0])
        # combined_ndet_SSE = combined_ndet_SSE + ndet_SSE

    print(combined_nsim)

    p_val_R = combined_ndet_R / combined_nsim
    p_val_SSE = combined_ndet_SSE / combined_nsim
    print('Using R: there were '+str(combined_ndet_R)+' false detections in '+str(combined_nsim)+' trials, which results in a p-value of '+str(p_val_R))
    print('Using SSE: there were '+str(combined_ndet_SSE)+' false detections in '+str(combined_nsim)+' trials, which results in a p-value of '+str(p_val_SSE))

    np.save('TK_results/all_TK_'+obs+'_'+str(emin)+'-'+str(emax)+'keV_'+str(tbin)+'s_'+psd_model+'.npy', 
            np.array([combined_ndet_R, p_val_R, combined_ndet_SSE, p_val_SSE, combined_nsim]))

if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser(description='Analyze the result of TK simulations.')

    # Adding arguments
    parser.add_argument('--obsid', type=str, help='Observation ID')
    parser.add_argument('--emin', type=float, default=1, help='Minimum energy of the band you are considering (default: 1)')
    parser.add_argument('--emax', type=float, default=4, help='Maximum energy of the band you are considering (default: 4)')
    parser.add_argument('--tbin', type=int, default=20, help='Time bin size in seconds (default: 20)')
    parser.add_argument('--psd_model', type=str, default='broken_powerlaw', help='PSD model (default: broken_powerlaw)')

    # Parse arguments
    args = parser.parse_args()

    combine_TK_results(obs=args.obsid, emin=args.emin, emax=args.emax, tbin=args.tbin, psd_model=args.psd_model)
