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

# fig_path = './figures/'
# data_path = '../data_paper2/xmm/lightcurves/'
save_path = '/pool001/mmasters/' # if /pool001/ is working on engaging
# save_path = './save_tmp/' # in home directory, for if /pool001/ is not working on engaging
mcmc_path = save_path+'broadband_mcmcs/' # if /pool001/ is working on engaging
# mcmc_path = './broadband_mcmcs_tmp/' # in home directory, for if /pool001/ is not working on engaging

# binned FFTs
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

########################################################################
################################ MODELS ################################
########################################################################

# simple power law
def powerlaw(v, N0, b, c):
    return N0 * v**(-b) + c
# lorentzian
def lorentzian(v, R, delta, c):
    return ( 2 * R**2 * delta / np.pi / (delta**2 + v**2)) + c

########################################################################
############################# SET UP BAYES #############################
########################################################################

# first we need to fit the continuum model -- then we will use this to generate simulated light curves
def log_likelihood(theta, f, y, model):
    if model == 'powerlaw':
        N0, b, c = theta
        model_y = powerlaw(f, N0, b, c)
    elif model == 'lorentzian':
        R, delta, c = theta
        model_y = lorentzian(f, R, delta, c)
    return 2 * np.sum((y / model_y) + np.log(model_y))

def log_prior(theta, model):
    if model == 'powerlaw':
        N0, b, c = theta
        if 0 < N0 < 1e10 and -5 < b < 5 and 0 < c < 100:
            return 0.0
    elif model == 'lorentzian':
        R, delta, c = theta
        if 0 < R < 1e2 and 1e-10 < delta < 1e1 and 1e-10 < c < 100:
            return 0.0
    return -np.inf

def log_probability(theta, f, y, model):
    lp = log_prior(theta, model)
    if not np.isfinite(lp):
        return -np.inf
    return lp - log_likelihood(theta, f, y, model)

########################################################################
############################### MAKE PSD ###############################
########################################################################

def get_key_values(obs, data_path, emin, emax, tbin=20):

    lc_file = "PN_"+obs+"_"+str(emin)+"-"+str(emax)+"_"+str(tbin)+"s.lc"
    hdu = fits.open(data_path+lc_file)
    data = hdu[1].data
    data = data[~np.isnan(data['RATE'])]
    time = data['TIME']-data['TIME'][0]
    rate = data['RATE']
    err = data['ERROR']

    N = len(time)
    mean = np.mean(rate)
    rms = np.sqrt(np.sum((rate - mean)**2) / (N - 1))

    return N, mean, rms

def make_psd(obs, data_path, emin, emax, tbin=20, n=10):

    if isinstance(obs, str):

        lc_file = "PN_"+obs+"_"+str(emin)+"-"+str(emax)+"_"+str(tbin)+"s.lc"
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
            return x, y, f_min, f_max

        # bin the data
        mean_bin_edges, mean_bin_values = bin_fft_data(x, y, n, 'mean')
        std_bin_edges, std_bin_values = bin_fft_data(x, y, n, 'std') / np.sqrt(n)
        bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
        bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])
        freq, power, freq_err, power_err = bin_midpoints, mean_bin_values, bin_widths, std_bin_values
        
    else:

        # save a big array of freqs + powers for all of the obs
        freqs = np.array([])
        powers = np.array([])
        
        # loop over all obs
        for i,obsid in enumerate(obs):

            lc_file = "PN_"+obsid+"_"+str(emin)+"-"+str(emax)+"_"+str(tbin)+"s.lc"
            hdu = fits.open(data_path+lc_file)
            data = hdu[1].data
            data = data[~np.isnan(data['RATE'])]
            time = data['TIME']-data['TIME'][0]
            rate = data['RATE']
            err = data['ERROR']

            if np.where(np.diff(time) != tbin)[0].size != 0:
                print('FOR OBSID '+obsid+': YOU HAVE A SERIOUS PROBLEM. YOUR DATA IS NOT CONTINUOUS. YOU ARE LINEARLY INTERPOLATING THE GAPS. MAKE SURE THAT THESE ARE NOT TOO LONG.')

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

            freqs = np.concatenate((freqs, x))
            powers = np.concatenate((powers, y))

        # bin the data once we've looped over all obs
        mean_bin_edges, mean_bin_values = bin_fft_data(freqs, powers, n, 'mean')
        std_bin_edges, std_bin_values = bin_fft_data(freqs, powers, n, 'std') / np.sqrt(n)
        bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
        bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])
        freq, power, freq_err, power_err = bin_midpoints, mean_bin_values, bin_widths, std_bin_values

        # compute min and max frequency now that we've done the binning
        f_min = freq[0] - freq_err[0]
        f_max = freq[-1] + freq_err[-1]

    return freq, power, freq_err, power_err, f_min, f_max

########################################################################
############################## MLE + MCMC ##############################
########################################################################

def MLE_and_MCMC(obs, data_path, emin, emax, tbin=20, n=0, nwalkers=32, use_model='all', nsteps=5500, nburn=500):

    # run the initial fit
    freq, power, f_min, f_max = make_psd(obs, data_path, emin, emax, tbin, n)

    # first maximize likelihood - power law
    if use_model == 'powerlaw' or use_model == 'all':
        ndim_pl = 3
        initial_guess_pl = [1e-4, 2, 1]
        result_pl = opt.minimize(log_likelihood, initial_guess_pl, args=(freq, power, 'powerlaw'),
                                bounds=((1e-8,1e4),(-2,2),(1e-2,1e2)))
        maxlike_pl = result_pl.x
        print(maxlike_pl)
        p0_pl = maxlike_pl * (1 + 1e-3 * np.random.randn(nwalkers, ndim_pl))

    # first maximize likelihood - broken power law
    if use_model == 'lorentzian' or use_model == 'all':
        ndim_lor = 3
        initial_guess_lor = [5e-1, 1e-3, 1]
        result_lor = opt.minimize(log_likelihood, initial_guess_lor, args=(freq, power, 'lorentzian'),
                                  bounds=((0,100),(1e-6,1),(1e-4,100)))
        maxlike_lor = result_lor.x
        print(maxlike_lor)
        p0_lor = maxlike_lor * (1 + 1e-3 * np.random.randn(nwalkers, ndim_lor))

    # make a nice figure showing the fits and residuals for both models
    if use_model == 'all':
        fig, axs = plt.subplots(figsize=(20,22), nrows=3, gridspec_kw={'hspace':0, 'height_ratios':[1,0.3,0.3]})
        ax = axs[0]
        grid = np.linspace(f_min, f_max, 1000)
        ax.step(freq, power, color='k', lw=2, where='mid')
        ax.plot(grid, powerlaw(grid, *maxlike_pl), color='xkcd:peach', lw=5, ls='--', label='Maximum Likelihood Power-Law')
        ax.plot(grid, lorentzian(grid, *maxlike_lor), color='slateblue', lw=5, ls='--', label='Maximum Likelihood Lorentzian')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Periodogram')
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(8e-2,3e2)
        ax = axs[1]
        ax.axhline(1, color='xkcd:peach', ls='--', lw=5)
        ax.step(freq, power / powerlaw(freq, *maxlike_pl), color='k', lw=2, where='mid')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('D / M')
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(3e-2,2e1)
        ax = axs[2]
        ax.axhline(1, color='slateblue', ls='--', lw=5)
        ax.step(freq, power / lorentzian(freq, *maxlike_lor), color='k', lw=2, where='mid')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('D / M')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(3e-2,2e1)

    # make a nice figure showing the fits and residuals for power-law model
    if use_model == 'powerlaw':
        fig, axs = plt.subplots(figsize=(20,14), nrows=2, gridspec_kw={'hspace':0, 'height_ratios':[1,0.3]})
        ax = axs[0]
        grid = np.linspace(f_min, f_max, 1000)
        ax.step(freq, power, color='k', lw=2, where='mid')
        ax.plot(grid, powerlaw(grid, *maxlike_pl), color='xkcd:peach', lw=5, ls='--', label='Maximum Likelihood Power-Law')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Periodogram')
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(8e-2,3e2)
        ax = axs[1]
        ax.axhline(1, color='xkcd:peach', ls='--', lw=5)
        ax.step(freq, power / powerlaw(freq, *maxlike_pl), color='k', lw=2, where='mid')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('D / M')
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(3e-2,2e1)

    # make a nice figure showing the fits and residuals for lorentzian model
    if use_model == 'lorentzian':
        fig, axs = plt.subplots(figsize=(20,14), nrows=2, gridspec_kw={'hspace':0, 'height_ratios':[1,0.3]})
        ax = axs[0]
        grid = np.linspace(f_min, f_max, 1000)
        ax.step(freq, power, color='k', lw=2, where='mid')
        ax.plot(grid, lorentzian(grid, *maxlike_lor), color='slateblue', lw=5, ls='--', label='Maximum Likelihood Lorentzian')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Periodogram')
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(8e-2,3e2)
        ax = axs[1]
        ax.axhline(1, color='slateblue', ls='--', lw=5)
        ax.step(freq, power / lorentzian(freq, *maxlike_lor), color='k', lw=2, where='mid')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('D / M')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(3e-2,2e1)

    # grab n_rand samples and pick out the median fit
    n_rand = 100
    if use_model == 'powerlaw' or use_model == 'all':

        # mcmc file name (should have been run in broadband.ipynb)
        fname = mcmc_path+'broadband_'+obs+'_'+str(emin)+'-'+str(emax)+'keV_mcmc_powerlaw.h5'

        # you should have run the MCMC already - load it in here
        if os.path.exists(fname):
            backend = emcee.backends.HDFBackend(fname, read_only=True)
            samples_pl = backend.get_chain(discard=nburn, flat=True)
            Dmin_H0 = min(-backend.get_log_prob(discard=nburn, flat=True))
        # if we haven't run the MCMC yet, you have a problem
        else:
            raise FileNotFoundError
            # do not actually run the MCMC - if it can't find it you have an issue

        # make a quick corner plot to check
        lab = [r"$\log N_0$", r"$\alpha$", r"$c$"]
        fig = corner.corner(samples_pl, labels=lab, truths=maxlike_pl, figsize=(18,18), labelpad=1)

        # grab best fit MCMC
        fit_pl = np.zeros((len(grid),n_rand))
        fit_binned_pl = np.zeros((len(freq),n_rand))
        for i in range(n_rand):
            use_N0, use_b, use_c = samples_pl[i,:]
            fit_pl[:,i] = powerlaw(grid, use_N0, use_b, use_c)
            fit_binned_pl[:,i] = powerlaw(freq, use_N0, use_b, use_c)
        med_fit_pl = np.median(fit_pl, axis=1)
        med_fit_binned_pl = np.median(fit_binned_pl, axis=1)
        axs[0].plot(grid, med_fit_pl, color='darkblue', lw=5, ls='--', label='Median Power-Law (MCMC)')

        # find T_R
        Rhat_pl = 2 * power / med_fit_binned_pl #null(bin_midpoints, *popt_null)
        Rhat_max_pl = np.max(Rhat_pl)
        QPO_freq_pl = freq[np.argmax(Rhat_pl)]
        print('R (power-law): ', Rhat_max_pl)
        print('QPO freq. (power-law): ', QPO_freq_pl)

        # find T_SSE
        SSE_pl = np.sum(((power - med_fit_binned_pl) / med_fit_binned_pl)**2)
        print('SSE (power-law): ', SSE_pl)

    if use_model == 'lorentzian' or use_model == 'all':

        # mcmc file name (should have been run in broadband.ipynb)
        fname = mcmc_path+'broadband_'+obs+'_'+str(emin)+'-'+str(emax)+'keV_mcmc_lorentzian.h5'

        # you should have run the MCMC already - load it in here
        if os.path.exists(fname):
            backend = emcee.backends.HDFBackend(fname, read_only=True)
            samples_lor = backend.get_chain(discard=nburn, flat=True)
            Dmin_H2 = min(-backend.get_log_prob(discard=nburn, flat=True)) 
        # if we haven't run the MCMC yet, you have a problem
        else:
            raise FileNotFoundError
            # do not actually run the MCMC - if it can't find it you have an issue
            # this is an example of how you would run the MCMC, if you hadn't already run it before:
            # backend.reset(nwalkers, ndim_lor)
            # sampler_lor = emcee.EnsembleSampler(nwalkers, ndim_lor, log_probability, args=(freq, power, 'lorentzian'), backend=backend)
            # sampler_lor.run_mcmc(p0_lor, nsteps, progress=True)
            # samples_lor = sampler_lor.get_chain(discard=nburn, flat=True)
            # Dmin_H2 = min(-sampler_lor.get_log_prob(discard=nburn, flat=True)) 

        # make a quick corner plot to check
        lab = [r"$R$", r"$\Delta$", r"$c$"]
        fig = corner.corner(samples_lor, labels=lab, truths=maxlike_lor, figsize=(18,18), labelpad=1)

        # grab best fit MCMC
        fit_lor = np.zeros((len(grid),n_rand))
        fit_binned_lor = np.zeros((len(freq),n_rand))
        for i in range(n_rand):
            use_R, use_delta, use_c = samples_lor[i,:]
            fit_lor[:,i] = lorentzian(grid, use_R, use_delta, use_c)
            fit_binned_lor[:,i] = lorentzian(freq, use_R, use_delta, use_c)
        med_fit_lor = np.median(fit_lor, axis=1)
        med_fit_binned_lor = np.median(fit_binned_lor, axis=1)
        axs[0].plot(grid, med_fit_lor, color='mediumseagreen', lw=5, ls='--', label='Median Lorentzian (MCMC)')

        # find T_R
        Rhat_lor = 2 * power / med_fit_binned_lor 
        Rhat_max_lor = np.max(Rhat_lor)
        QPO_freq_lor = freq[np.argmax(Rhat_lor)]
        print('R (Lorentzian): ', Rhat_max_lor)
        print('QPO freq. (Lorentzian): ', QPO_freq_lor)

        # find T_SSE
        SSE_lor = np.sum(((power - med_fit_binned_lor) / med_fit_binned_lor)**2)
        print('SSE (Lorentzian): ', SSE_lor)

    axs[0].legend(fontsize=24, loc='lower left')

    # for LRT + saving all the samples
    if use_model == 'all':

        # power law values to save
        save_N0_pl = samples_pl[:,0]
        save_b_pl = samples_pl[:,1]
        save_c_pl = samples_pl[:,2]

        # lorentzian values to save
        save_R_lor = samples_lor[:,0]
        save_delta_lor = samples_lor[:,1]
        save_c_lor = samples_lor[:,2]

        return save_N0_pl, save_b_pl, save_c_pl, Rhat_max_pl, QPO_freq_pl, SSE_pl, \
        save_R_lor, save_delta_lor, save_c_lor, Rhat_max_lor, QPO_freq_lor, SSE_lor
    
    elif use_model == 'powerlaw':
        return Rhat_max_pl, QPO_freq_pl, SSE_pl
    elif use_model == 'lorentzian':
        return Rhat_max_lor, QPO_freq_lor, SSE_lor
    
########################################################################
############################ TIMMER & KONIG ############################
########################################################################
    
def timmerkonig_sims(duration, tbin, psd_model, psd_params, mean_obs, rms_obs, plot=False):
    """
    Simulate a light curve with power-law noise from the methodology outlined in Timmer & Konig (1995).

    Parameters:
    duration (float): Total duration of the light curve in seconds.
    tbin (float): Time binning (sampling interval) in seconds.
    psd_model (str): Model to use for the PSD.
    psd_params (array): Array of model parameters to pass to the PSD.
    mean_obs (float): mean rate of the observed data to match simulations to
    rms_obs (float): rms of the observed data to match simulations to
    plot (bool): whether or not to plot the resulting light curve

    Returns:
    np.array: Simulated light curve.
    """
    # number of samples
    N = int(duration / tbin)

    # frequency array
    all_freqs = np.fft.fftfreq(N, d=tbin)
    pos_freqs = all_freqs[all_freqs > 0] # exclude 0

    # compute gaussian distributed random numbers for the real and imaginary parts of the phase
    np.random.seed(seed=np.random.randint(0,10000))
    rand_re = np.random.normal(size=pos_freqs.shape)
    rand_im = np.random.normal(size=pos_freqs.shape)
    rand_im[-1] = 0 # nyquist freq is real
    sim_spec = rand_re + 1j * rand_im
    if psd_model == 'powerlaw':
        psd_shape = powerlaw(pos_freqs, psd_params[0], psd_params[1], 0)
    elif psd_model == 'lorentzian':
        psd_shape = lorentzian(pos_freqs, psd_params[0], psd_params[1], 0)
    sim_spec = sim_spec * np.sqrt(0.5 * psd_shape) # for a more general power spectrum
    sim_spec[0] = 0 + 0j

    # add in the negative frequencies 
    sim_spec_neg = np.conj(sim_spec[1:-1]) # ignore the 0 freq and the nyquist frequency
    sim_spec_neg = sim_spec_neg[::-1] # reverse the order
    full_spec = np.concatenate([sim_spec, sim_spec_neg])

    # inverse FFT to get the time series
    light_curve = np.fft.irfft(full_spec, n=N)

    # scale to the observations
    mean_sim = np.mean(light_curve)
    rms_sim = np.sqrt(np.sum((light_curve - mean_sim)**2) / (N - 1))
    light_curve = light_curve * rms_obs / rms_sim # scale the rms
    light_curve = (light_curve - mean_sim) + mean_obs 

    # add noise, based on the sqrt of the constant white noise component from the original fit to the PSD
    noise_max = np.sqrt(psd_params[2] / (2 * tbin)) # need to account for the power / freq, i.e. multiply by the nyquist freq
    noise = np.random.normal(0, noise_max, size=len(light_curve))
    light_curve = light_curve + noise

    # times for plotting
    times = np.arange(0,duration,tbin)

    if plot:

        fig, ax = plt.subplots(figsize=(18,10))
        ax.plot(times, light_curve, 'k-')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Rate [cps]')

    return light_curve

def make_TK_psd(lc, tbin, psd_model, psd_params, n=0, plot=False):

    # length of the light curve will help with duration
    N_tot = len(lc)
    duration = N_tot * tbin

    # time array
    lc_time = np.arange(0, duration, tbin)

    # poisson noise
    lc_err = np.sqrt(lc * tbin)

    # maximum frequency that can be probed = nyquist = 1 / (2 * tbin)
    f_max = 1 / (2 * tbin)
    # minimimum frequency that can be probed = 1 / T_obs
    T_obs = max(lc_time)
    f_min = 1 / T_obs
    lc = pylag.LightCurve(t=lc_time, r=lc, e=lc_err, interp_gaps=True) # there shouldn't be any gaps though
    per = pylag.Periodogram(lc)

    x = per.freq
    y = per.periodogram
    # remove the 0 frequency
    x = x[1:]
    y = y[1:]

    if n == 0:
        # there will be no error bars on the data
        freq, power = x, y

    else:
        # bin the data
        mean_bin_edges, mean_bin_values = bin_fft_data(x, y, n, 'mean')
        std_bin_edges, std_bin_values = bin_fft_data(x, y, n, 'std') / np.sqrt(n)
        bin_midpoints = 0.5 * (mean_bin_edges[1:] + mean_bin_edges[:-1])
        bin_widths = 0.5 * (mean_bin_edges[1:] - mean_bin_edges[:-1])
        freq, power, freq_err, power_err = bin_midpoints, mean_bin_values, bin_widths, std_bin_values

    if plot:

        # plot the power spectrum, and compare to the input
        fig, ax = plt.subplots(figsize=(14,10))
        if n == 0:
            ax.step(freq, power, 'k', lw=2, label='Simulated')
        else:
            ax.errorbar(freq, power, xerr=freq_err, yerr=power_err, fmt='o', ms=0, color='k', capsize=0, label='Simulated')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Periodogram')
        
        freq_grid = np.linspace(f_min, f_max, 1000)
        if psd_model == 'powerlaw':
            psd_input = powerlaw(freq_grid, psd_params[0], psd_params[1], 0)
            psd_total = powerlaw(freq_grid, *psd_params)
        elif psd_model == 'lorentzian':
            psd_input = lorentzian(freq_grid, psd_params[0], psd_params[1], 0)
            psd_total = lorentzian(freq_grid, *psd_params)

        ax.plot(freq_grid, psd_input, 'darkorange', ls='--', lw=3, label='Input')
        ax.plot(freq_grid, psd_total, 'slateblue', ls='--', lw=3, label='Observed')
        ax.legend()

    if n == 0:
        return freq, power
    else:
        return freq, power, freq_err, power_err
    
def fit_TK(ind, taskid, obs, emin, emax, tbin, freq, power, freq_err=None, power_err=None, psd_model='powerlaw', psd_params=[1e-2, 1, 1], 
           nwalkers=32, nsteps=5500, nburn=500, plot=False, maxlike_init=True, SSE_input=0):

    # perform MLE 
    ndim = len(psd_params) # should work for either power law or lorentzian model
    if psd_model == 'powerlaw':
        bounds = ((0,1e4),(0,5),(0,1e2))
        bounds_start = ((0,0.2),(0,2),(0,1))
    if psd_model == 'lorentzian':
        bounds = ((0,100),(1e-6,1),(1e-4,100))
    result = opt.minimize(log_likelihood, psd_params, args=(freq, power, psd_model), bounds=bounds)
    maxlike = result.x
    if maxlike_init:
        p0 = maxlike * (1 + 1e-3 * np.random.randn(nwalkers, ndim))
    else:
        # use parameter bounds and create a uniformly sampled set of walkers to start, if not running with MLE to start
        p0 = np.random.rand(nwalkers, ndim)  # random values in [0, 1)
        for i in range(ndim):
            p0[:,i] = p0[:,i] * (bounds_start[i][1] - bounds_start[i][0]) + bounds_start[i][0]  # scale and shift to parameter bounds

    # then MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(freq, power, psd_model))
    sampler.run_mcmc(p0, nsteps, progress=False)
    samples = sampler.get_chain(discard=nburn, flat=True)

    f_min = min(freq)
    f_max = max(freq)
    grid = np.linspace(f_min, f_max, 1000)
    
    # grab n_rand samples and pick out the median fit from the MCMC
    n_rand = 100
    fit = np.zeros((len(grid),n_rand))
    fit_binned = np.zeros((len(freq),n_rand))
    for i in range(n_rand):
        if psd_model == 'powerlaw':
            use_N0, use_b, use_c, = samples[i,:]
            fit[:,i] = powerlaw(grid, use_N0, use_b, use_c)
            fit_binned[:,i] = powerlaw(freq, use_N0, use_b, use_c)
        if psd_model == 'lorentzian':
            use_R, use_delta, use_c = samples[i,:]
            fit[:,i] = lorentzian(grid, use_R, use_delta, use_c)
            fit_binned[:,i] = lorentzian(freq, use_R, use_delta, use_c)
    med_fit = np.median(fit, axis=1)
    med_fit_binned = np.median(fit_binned, axis=1)

    # find T_R
    Rhat_TK = 2 * power / med_fit_binned 
    Rhat_max_TK = max(Rhat_TK)

    # find T_SSE
    SSE_TK = np.sum(((power - med_fit_binned) / med_fit_binned)**2)

    # only plot if this is a "bad" fit -- i.e. if this simulated PSD gives a worse SSE than what we found for the observed data
    if plot and (SSE_TK > SSE_input):

        if not os.path.exists(save_path+'save_TK_fits/'):
            os.mkdir(save_path+'save_TK_fits/')

        # not in log space
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(grid, med_fit, color='dodgerblue', lw=5, ls='--', label='Median Fit from MCMC')
        if not freq_err is None:
            ax.errorbar(freq, power, xerr=freq_err, yerr=power_err, fmt='o', ms=0, color='k', capsize=0, label='Simulated Data')
        else:
            ax.step(freq, power, 'k-', lw=2, label='Simulated Data')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Periodogram')
        ax.legend(fontsize=24)
        plt.savefig(save_path+'save_TK_fits/mcmc_fit_'+obs+'_'+str(emin)+'-'+str(emax)+'keV_'+str(tbin)+'s_'+psd_model+'_'+str(ind)+'_'+str(taskid)+'.png', bbox_inches='tight')
        plt.close()

        # not in log space
        fig, axs = plt.subplots(nrows=ndim, figsize=(10,15), gridspec_kw={'hspace':0, 'wspace':0})
        for i in range(ndim):
            ax = axs[i]
            ax.plot(np.linspace(0,len(samples[:,i]),len(samples[:,i])), samples[:,i], 'k-', lw=0.1)
            if i == (ndim-1):
                ax.set_xlabel('Steps')
        plt.savefig(save_path+'save_TK_fits/mcmc_convergence_'+obs+'_'+str(emin)+'-'+str(emax)+'keV_'+str(tbin)+'s_'+psd_model+'_'+str(ind)+'_'+str(taskid)+'.png', bbox_inches='tight')
        plt.close()

    return Rhat_max_TK, SSE_TK

def run_TK_sims_and_analysis(taskid, obs, data_path, emin, emax, tbin, n=0, n_sim=100, n_jobs=16, psd_model='broken_powerlaw', maxlike_init=True):

    # get the necessary parameters from the MCMC
    save_N0_pl, save_b_pl, save_c_pl, Rhat_max_pl, QPO_freq_pl, SSE_pl, save_R_lor, save_delta_lor, save_c_lor, Rhat_max_lor, QPO_freq_lor, SSE_lor = \
            MLE_and_MCMC(obs, data_path, emin, emax, tbin, n, nsteps=55000, nburn=5000)
    N, mean, rms = get_key_values(obs, data_path, emin, emax, tbin)
    dur = tbin * N

    # Rhat, SSE to compare to
    if psd_model == 'powerlaw':
        Rhat_comp = Rhat_max_pl
        SSE_comp = SSE_pl
    elif psd_model == 'lorentzian':
        Rhat_comp = Rhat_max_lor
        SSE_comp = SSE_lor

    # function to perform a TK simulation, fit it, and compare statistics - this is what will be run in parallel
    def run_parallel(i):

        if psd_model == 'powerlaw':
            rand_int = np.random.randint(0,len(save_N0_pl))
            sim_N0 = save_N0_pl[rand_int]
            sim_b = save_b_pl[rand_int]
            sim_c = save_c_pl[rand_int]
            psd_params = [sim_N0, sim_b, sim_c]
        if psd_model == 'lorentzian':
            rand_int = np.random.randint(0,len(save_R_lor))
            sim_R = save_R_lor[rand_int]
            sim_delta = save_delta_lor[rand_int]
            sim_c = save_c_lor[rand_int]
            psd_params = [sim_R, sim_delta, sim_c]

        # run simulation and fit the new psd
        lc = timmerkonig_sims(dur, tbin, psd_model, psd_params, mean, rms)
        freq, power = make_TK_psd(lc, tbin, psd_model, psd_params, plot=False)
        Rhat_TK, SSE_TK = fit_TK(i, taskid, obs, emin, emax, tbin, freq, power, psd_model=psd_model, psd_params=psd_params, plot=True, maxlike_init=maxlike_init, SSE_input=SSE_comp)

        return Rhat_TK, SSE_TK

    # run the above function in parallel with n_jobs
    results = Parallel(n_jobs=n_jobs)(delayed(run_parallel)(i) for i in range(n_sim))
    Rhat_TK = [x[0] for x in results]
    SSE_TK = [x[1] for x in results]

    # create directory in which to save results
    if not os.path.exists(save_path+'TK_results/'):
        os.mkdir(save_path+'TK_results/')

    # save the results
    np.save(save_path+'TK_results/Rhat_TK_'+obs+'_'+str(emin)+'-'+str(emax)+'keV_'+str(tbin)+'s_'+psd_model+'_'+str(taskid)+'.npy', Rhat_TK)
    np.save(save_path+'TK_results/SSE_TK_'+obs+'_'+str(emin)+'-'+str(emax)+'keV_'+str(tbin)+'s_'+psd_model+'_'+str(taskid)+'.npy', SSE_TK)
    n_fd = len(np.where(Rhat_TK > Rhat_comp)[0])
    n_fd_SSE = len(np.where(SSE_TK > SSE_comp)[0])
    np.save(save_path+'TK_results/TK_'+obs+'_'+str(emin)+'-'+str(emax)+'keV_'+str(tbin)+'s_'+psd_model+'_'+str(taskid)+'.npy', np.array([Rhat_comp, n_fd, SSE_comp, n_fd_SSE, n_sim]))

    return Rhat_TK, Rhat_comp, SSE_TK, SSE_comp 

if __name__ == "__main__":

    # initialize parser
    parser = argparse.ArgumentParser(description='Run TK simulations and analysis.')

    # add arguments
    parser.add_argument('--obsid', type=str, help='Observation ID')
    parser.add_argument('--data_path', type=str, help='Path to the data directory')
    parser.add_argument('--emin', type=float, default=1, help='Minimum energy of the band you are considering (default: 1)')
    parser.add_argument('--emax', type=float, default=4, help='Maximum energy of the band you are considering (default: 4)')
    parser.add_argument('--tbin', type=int, default=20, help='Time bin size in seconds (default: 20)')
    parser.add_argument('--n', type=int, default=0, help='Number of bins for the power spectrum (default: 0)')
    parser.add_argument('--n_sim', type=int, default=10000, help='Number of simulations (default: 10000)')
    parser.add_argument('--n_jobs', type=int, default=100, help='Number of parallel jobs (default: 100)')
    parser.add_argument('--psd_model', type=str, default='powerlaw', help='PSD model (default: powerlaw). Only acceptable objects are powerlaw and lorentzian.')
    parser.add_argument('--maxlike_init', type=str, default='True', help='Whether to initialize MCMC walkers with maximum likelihood estimate (default: True)')

    # parse arguments
    args = parser.parse_args()

    # whether or not to start from the maximum likelihood estimate (for some reason this can be problematic for the power-law)
    if args.maxlike_init == 'False' or args.maxlike_init == '0' or args.maxlike_init == 'F':
        mli = False
    else:
        mli = True

    # call the function with parsed arguments
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0')) # this is just a # that the code will use to name things
    run_TK_sims_and_analysis(taskid=task_id, obs=args.obsid, data_path=args.data_path, emin=args.emin, emax=args.emax, tbin=args.tbin, n=args.n, 
                             n_sim=args.n_sim, n_jobs=args.n_jobs, psd_model=args.psd_model, maxlike_init=mli)
