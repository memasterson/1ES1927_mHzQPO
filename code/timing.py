# python file with functions to do timing analysis on 1ES 1927+654

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
import pandas as pd
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, NullFormatter
from matplotlib import ticker
import json
import glob
# from redshift get d_L
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
from scipy.stats import spearmanr, linregress
import pylag
from joblib import Parallel, delayed
from scipy.stats import binned_statistic, chi2

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

# function to bin up FFTs based on a certain number of frequencies per bin
def bin_fft_data(frequencies, power, n, statistic):
    """
    Bin the frequency and power data from FFT output.

    :param frequencies: Array of frequency values from FFT.
    :param power: Array of power values corresponding to each frequency.
    :param n: Number of points in each bin.
    :param statistic: Statistic to calculate ('mean' or 'std').
    :return: Bin edges and calculated statistic for each bin.
    """
    # Calculate the number of bins
    num_bins = len(frequencies) // n

    # Use binned_statistic to calculate the desired statistic for power
    bin_statistic, bin_edges, _ = binned_statistic(frequencies, power, statistic=statistic, bins=num_bins)

    return bin_edges, bin_statistic

# load in data and make a PSD, also returns data to make a light curve 
def make_psd(obs, data_path, emin, emax, tbin=20, n=10, tmin=None, tmax=None):

    if isinstance(obs, str):

        lc_file = "PN_"+obs+"_"+str(emin)+"-"+str(emax)+"_"+str(tbin)+"s.lc"
        hdu = fits.open(data_path+lc_file)
        data = hdu[1].data
        data = data[~np.isnan(data['RATE'])]
        time = data['TIME']-data['TIME'][0]
        rate = data['RATE']
        err = data['ERROR']

        if tmin is not None:
            use = time > tmin
            rate = rate[use]
            err = err[use]
            time = time[use]

        if tmax is not None:
            use = time < tmax
            rate = rate[use]
            err = err[use]
            time = time[use]

        if np.where(np.diff(time) != tbin)[0].size != 0:
            print('YOU HAVE A SERIOUS PROBLEM. YOUR DATA IS NOT CONTINUOUS. YOU ARE LINEARLY INTERPOLATING THE GAPS. MAKE SURE THAT THESE ARE NOT TOO LONG.')

        # maximum frequency that can be probed = nyquist = 1 / (2 * tbin)
        f_max = 1 / (2 * tbin)
        # minimimum frequency that can be probed = 1 / T_obs
        T_obs = max(time)
        f_min = 1 / T_obs
        lc = pylag.LightCurve(t=time, r=rate, e=err, interp_gaps=True)
        per = pylag.Periodogram(lc)

        x = per.freq
        y = per.periodogram
        # remove the 0 frequency
        x = x[1:]
        y = y[1:]

        if n == 0:
            return time, rate, err, x, y, 0, 0, f_min, f_max, 0, 0

        # bin the data
        mean_bin_edges, mean_bin_values = bin_fft_data(x, y, n, 'mean')
        std_bin_edges, std_bin_values = bin_fft_data(x, y, n, 'std')
        std_bin_values = std_bin_values / np.sqrt(n)
        bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
        bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])
        freq, power, freq_err, power_err = bin_midpoints, mean_bin_values, bin_widths, std_bin_values

        # for single observations, you do not need to keep track of start and true times
        tstarts = 0
        true_time = 0
        
    else:

        # save a big array of freqs + powers for all of the obs
        freqs = np.array([])
        powers = np.array([])
        time = np.array([])
        true_time = np.array([]) # for phase folding
        rate = np.array([])
        err = np.array([])
        tstarts = np.array([])
        
        avg_powers = None
        
        # loop over all obs
        for i,obsid in enumerate(obs):

            lc_file = "PN_"+obsid+"_"+str(emin)+"-"+str(emax)+"_"+str(tbin)+"s.lc"
            hdu = fits.open(data_path+lc_file)
            data = hdu[1].data
            data = data[~np.isnan(data['RATE'])]
            true_time_tmp = data['TIME']
            time_tmp = data['TIME']-data['TIME'][0]
            rate_tmp = data['RATE']
            err_tmp = data['ERROR']

            # save last time so we can add it to the time array
            if i == 0:
                t0 = 0
            else:
                t0 = time[-1]

            if tmax is not None:
                use = time_tmp < tmax
                rate_tmp = rate_tmp[use]
                err_tmp = err_tmp[use]
                time_tmp = time_tmp[use]
                true_time_tmp = true_time_tmp[use]

            tstarts = np.append(tstarts, t0)
            time = np.append(time, time_tmp+t0)
            rate = np.append(rate, rate_tmp)
            err = np.append(err, err_tmp)
            true_time = np.append(true_time, true_time_tmp)

            if np.where(np.diff(time_tmp) != tbin)[0].size != 0:
                print('FOR OBSID '+obsid+': YOU HAVE A SERIOUS PROBLEM. YOUR DATA IS NOT CONTINUOUS. YOU ARE LINEARLY INTERPOLATING THE GAPS. MAKE SURE THAT THESE ARE NOT TOO LONG.')

            # maximum frequency that can be probed = nyquist = 1 / (2 * tbin)
            f_max = 1 / (2 * tbin)
            # minimimum frequency that can be probed = 1 / T_obs
            T_obs = max(time_tmp)
            f_min = 1 / T_obs
            lc = pylag.LightCurve(t=time_tmp, r=rate_tmp, e=err_tmp, interp_gaps=True)
            per = pylag.Periodogram(lc)

            x = per.freq
            y = per.periodogram
            # remove the 0 frequency
            x = x[1:]
            y = y[1:]

            freqs = np.concatenate((freqs, x))
            powers = np.concatenate((powers, y))

            if tmax is not None:
                if avg_powers is None:
                    avg_powers = np.zeros(len(powers))
                    avg_freqs = x
                avg_powers += y

        # if tmax is not None, then we want to average at each frequency point (the individual obs should have the same frequency arrays)
        if tmax is not None:
            avg_powers = avg_powers / len(obs)
            # bin the data once we've looped over all obs
            mean_bin_edges, mean_bin_values = bin_fft_data(avg_freqs, avg_powers, n, 'mean')
            std_bin_edges, std_bin_values = bin_fft_data(avg_freqs, avg_powers, n, 'std') 
            std_bin_values = std_bin_values / np.sqrt(n * len(obs))
            bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
            bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])
            freq, power, freq_err, power_err = bin_midpoints, mean_bin_values, bin_widths, std_bin_values

        else:
            # bin the data once we've looped over all obs
            mean_bin_edges, mean_bin_values = bin_fft_data(freqs, powers, n, 'mean')
            std_bin_edges, std_bin_values = bin_fft_data(freqs, powers, n, 'std') 
            std_bin_values = std_bin_values / np.sqrt(n)
            bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
            bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])
            freq, power, freq_err, power_err = bin_midpoints, mean_bin_values, bin_widths, std_bin_values
            
        # compute min and max frequency now that we've done the binning
        f_min = freq[0] - freq_err[0]
        f_max = freq[-1] + freq_err[-1]

    return time, rate, err, freq, power, freq_err, power_err, f_min, f_max, tstarts, true_time

# same thing, but for MOS data
def make_MOS_psd(obs, data_path, emin, emax, tbin=20, n=10):

    if isinstance(obs, str):

        lc_file = "MOS_"+obs+"_"+str(emin)+"-"+str(emax)+"_"+str(tbin)+"s.lc"
        hdu = fits.open(data_path+lc_file)
        data = hdu[1].data
        data = data[~np.isnan(data['RATE'])]
        time = data['TIME']-data['TIME'][0]
        rate = data['RATE']
        err = data['ERROR']

        if np.where(np.diff(time) != tbin)[0].size != 0:
            print('YOU HAVE A SERIOUS PROBLEM. YOUR DATA IS NOT CONTINUOUS. YOU ARE LINEARLY INTERPOLATING THE GAPS. MAKE SURE THAT THESE ARE NOT TOO LONG.')

        # maximum frequency that can be probed = nyquist = 1 / (2 * tbin)
        f_max = 1 / (2 * tbin)
        # minimimum frequency that can be probed = 1 / T_obs
        T_obs = max(time)
        f_min = 1 / T_obs
        lc = pylag.LightCurve(t=time, r=rate, e=err, interp_gaps=True)
        per = pylag.Periodogram(lc)

        x = per.freq
        y = per.periodogram
        # remove the 0 frequency
        x = x[1:]
        y = y[1:]

        if n == 0:
            return time, rate, err, x, y, 0, 0, f_min, f_max, 0, 0

        # bin the data
        mean_bin_edges, mean_bin_values = bin_fft_data(x, y, n, 'mean')
        std_bin_edges, std_bin_values = bin_fft_data(x, y, n, 'std')
        std_bin_values = std_bin_values / np.sqrt(n)
        bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
        bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])
        freq, power, freq_err, power_err = bin_midpoints, mean_bin_values, bin_widths, std_bin_values

        # for single observations, you do not need to keep track of start and true times
        tstarts = 0
        true_time = 0
        
    else:

        # save a big array of freqs + powers for all of the obs
        freqs = np.array([])
        powers = np.array([])
        time = np.array([])
        true_time = np.array([]) # for phase folding
        tstarts = np.array([])
        rate = np.array([])
        err = np.array([])
        
        # loop over all obs
        for i,obsid in enumerate(obs):

            lc_file = "MOS_"+obsid+"_"+str(emin)+"-"+str(emax)+"_"+str(tbin)+"s.lc"
            hdu = fits.open(data_path+lc_file)
            data = hdu[1].data
            data = data[~np.isnan(data['RATE'])]
            time_tmp = data['TIME']-data['TIME'][0]
            if i == 0:
                t0 = 0
            else:
                t0 = time[-1]
            time = np.append(time, time_tmp+t0)
            rate_tmp = data['RATE']
            rate = np.append(rate, rate_tmp)
            err_tmp = data['ERROR']
            err = np.append(err, err_tmp)
            tstarts = np.append(tstarts, t0)
            true_time = np.append(true_time, data['TIME'])

            if np.where(np.diff(time_tmp) != tbin)[0].size != 0:
                print('FOR OBSID '+obsid+': YOU HAVE A SERIOUS PROBLEM. YOUR DATA IS NOT CONTINUOUS. YOU ARE LINEARLY INTERPOLATING THE GAPS. MAKE SURE THAT THESE ARE NOT TOO LONG.')

            # maximum frequency that can be probed = nyquist = 1 / (2 * tbin)
            f_max = 1 / (2 * tbin)
            # minimimum frequency that can be probed = 1 / T_obs
            T_obs = max(time_tmp)
            f_min = 1 / T_obs
            lc = pylag.LightCurve(t=time_tmp, r=rate_tmp, e=err_tmp, interp_gaps=True)
            per = pylag.Periodogram(lc)

            x = per.freq
            y = per.periodogram
            # remove the 0 frequency
            x = x[1:]
            y = y[1:]

            freqs = np.concatenate((freqs, x))
            powers = np.concatenate((powers, y))

        # bin the data once we've looped over all obs
        mean_bin_edges, mean_bin_values = bin_fft_data(freqs, powers, n, 'mean')
        std_bin_edges, std_bin_values = bin_fft_data(freqs, powers, n, 'std') 
        std_bin_values = std_bin_values / np.sqrt(n)
        bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
        bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])
        freq, power, freq_err, power_err = bin_midpoints, mean_bin_values, bin_widths, std_bin_values

        # compute min and max frequency now that we've done the binning
        f_min = freq[0] - freq_err[0]
        f_max = freq[-1] + freq_err[-1]

    return time, rate, err, freq, power, freq_err, power_err, f_min, f_max, tstarts, true_time

# function to phase-fold data
def phase_fold(obs, emin, emax, tbin, path, qpo_per, num_bins=20, max_time=None):

    lc_file = "PN_"+obs+"_"+str(emin)+"-"+str(emax)+"_"+str(tbin)+"s.lc"
    hdu = fits.open(path+lc_file)
    data = hdu[1].data
    data = data[~np.isnan(data['RATE'])]
    time = data['TIME']-data['TIME'][0]
    rate = data['RATE']
    err = data['ERROR']

    if max_time is not None:
        use_time = time < max_time
        time = time[use_time]
        rate = rate[use_time]

    # phase fold the light curve
    folded_times = time % qpo_per

    # bin
    num_pts = len(rate) // num_bins
    mean_bin_values, mean_bin_edges, _ = binned_statistic(folded_times, rate, statistic='mean', bins=num_bins)
    std_bin_values, std_bin_edges, _ = binned_statistic(folded_times, rate, statistic='std', bins=num_bins) #/ np.sqrt(num_bins)
    std_bin_values = std_bin_values / np.sqrt(num_pts)
    bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
    bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])

    return bin_midpoints, mean_bin_values, bin_widths, std_bin_values, folded_times, rate

# function to phase-fold data with multiple obsids
def phase_fold_multi(time, rate, qpo_per, num_bins=20):

    # phase fold the light curve
    folded_times = time % qpo_per

    # bin
    num_pts = len(rate) // num_bins
    mean_bin_values, mean_bin_edges, _ = binned_statistic(folded_times, rate, statistic='mean', bins=num_bins)
    std_bin_values, std_bin_edges, _ = binned_statistic(folded_times, rate, statistic='std', bins=num_bins) #/ np.sqrt(num_bins)
    std_bin_values = std_bin_values / np.sqrt(num_pts)
    bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
    bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])

    return bin_midpoints, mean_bin_values, bin_widths, std_bin_values, folded_times, rate

###################################################################
#### functions for fitting with various broadband noise models ####
###################################################################

# simple power law
def powerlaw(v, N0, b, c):
    return N0 * v**(-b) + c

# lorentzian
def lorentzian(v, R, delta, c):
    return ( 2 * R**2 * delta / np.pi / (delta**2 + v**2)) + c

def lorentzian_noconst(v, v0, R, delta):
    # using definition from Ingram+19 review
    # R ~ a0 in that review, delta and v0 are the same
    return ( R**2 / ((np.pi / 2) + np.arctan(v0 / delta)) ) * ( delta / (delta**2 + (v - v0)**2) )

# qpo with the power law broadband noise model
def qpo_pl(v, N0, b, c, v0, R, delta):
    return powerlaw(v, N0, b, c) + lorentzian_noconst(v, v0, R, delta)

# qpo with the lorenzian broadband noise model
def qpo_lor(v, R_bb, delta_bb, c, v0, R, delta):
    return lorentzian(v, R_bb, delta_bb, c) + lorentzian_noconst(v, v0, R, delta)