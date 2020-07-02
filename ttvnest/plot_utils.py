import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from scipy.ndimage import gaussian_filter as norm_kde
from .constants import *
from . import rebound_backend as rb
import matplotlib
import random
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20

def dynesty_plots(system, truthvals = None, outname = None):
	if system.results == None:
		raise ValueError("No retrieval found in your system object!")

	if outname is not None:
		names = [outname + '_cornerplot.png',
			outname + '_traceplot.png',
			outname + '_runplot.png']
	#cornerplot
	plt.figure(figsize = (20, 20))
	cfig, caxes = dyplot.cornerplot(system.results, color = 'blue',
		max_n_ticks = 3, labels = system.fit_param_names,
		truths = truthvals)
	if outname == None:
		plt.show()
	else:
		plt.savefig(names[0])
		plt.close('all')

	#traceplot
	plt.figure(figsize = (20, 20))
	tfig, taxes = dyplot.traceplot(system.results, truths = truthvals,
		show_titles = True, trace_cmap = 'viridis',
		labels = system.fit_param_names)
	if outname == None:
		plt.show()
	else:
		plt.savefig(names[1])
		plt.close('all')

	#runplot
	try:
		plt.figure(figsize = (20, 20))
		rfig, raxes = dyplot.runplot(system.results)
		if outname == None:
			plt.show()
		else:
			plt.savefig(names[2])
			plt.close('all')
			return names
	except ValueError:
		plt.close('all')
		print("Axis limits error on runplot; internal to dynesty")
		pass

	return None

def plot_linear_fit_results(system, uncertainty_curves = 0, sim_length = 2000,
	outname = None):
	if system.linear_fit_results == None:
		raise ValueError("No linear fit retrieval found in your "+
			"system object!")

	dresults = system.linear_fit_results
	samples = dresults.samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])
	ind = np.argmax(dresults.logl)
	max_like_res = dresults.samples[ind]

	samples_equal = dyfunc.resample_equal(samples, weights)
	inds = np.arange(samples_equal.shape[0])
	pick = np.random.choice(inds, uncertainty_curves, replace=False)
	random_samples = samples_equal[pick,:]

	#data and median result
	system.sim_length = sim_length
	models = system.linear_forward_model(max_like_res)

	unc_models = []
	for samp in random_samples:
		rand_models = system.linear_forward_model(samp)
		unc_models.append(rand_models)

	for i, planet in enumerate(system.planets):
		if planet.transiting:
			datum = planet.ttv
			err = planet.ttv_err
			epoch = planet.epochs
			model = models[i]
			mean_ephem = planet.mean_ephem

			ttv_datum = datum - mean_ephem(epoch)
			ttv_model = model - mean_ephem(epoch)

			plt.figure(figsize = (12, 8))
			plt.errorbar(epoch, ttv_datum*24.*60., 
				yerr = err*24.*60., linestyle = '', 
				color = 'k', marker = 'o', ms = 4, zorder = 1)
			plt.plot(epoch, ttv_model*24.*60.,
				color = 'blue', linewidth = 1, zorder = 1000)

			for unc_mod in unc_models:
				unc_model = unc_mod[i]
				unc_ttv_model = unc_model - mean_ephem(
					epoch)
				plt.plot(epoch,
					unc_ttv_model*24.*60., color = 'blue',
					linewidth = 0.5, alpha = 0.05,
					zorder = 10)
				plt.xlim(min(epoch)-1, max(epoch) + 1)
			plt.xlabel("Epoch")
			plt.ylabel("TTV [min]")
			if outname is None:
				plt.show()
				plt.close('all')
			else:
				plt.savefig(outname + f'_{i+1}_linearfit.png')
				plt.close('all')
	return None

def plot_results(system, uncertainty_curves = 0, sim_length = 2000,
	outname = None, print_max_likelihood_result = False):
	if system.results == None:
		raise ValueError("No retrieval found in your system object!")

	dresults = system.results
	samples = dresults.samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])
	ind = np.argmax(dresults.logl)
	max_like_res = dresults.samples[ind]

	samples_equal = dyfunc.resample_equal(samples, weights)
	inds = np.arange(samples_equal.shape[0])
	pick = np.random.choice(inds, uncertainty_curves, replace=False)
	random_samples = samples_equal[pick,:]

	#data and median result
	system.sim_length = sim_length
	models = system.forward_model(max_like_res)
	if print_max_likelihood_result:
		print(models)
	#uncertainty results
	unc_models = []
	for samp in random_samples:
		rand_models = system.forward_model(samp)
		unc_models.append(rand_models)

	for i, planet in enumerate(system.planets):
		if planet.transiting:
			datum = planet.ttv
			err = planet.ttv_err
			epoch = planet.epochs
			model = models[i]
			mean_ephem = planet.mean_ephem

			ttv_datum = datum - mean_ephem(epoch)
			ttv_model = model - mean_ephem(np.arange(len(model)))

			plt.figure(figsize = (12, 8))
			plt.errorbar(epoch, ttv_datum*24.*60., 
				yerr = err*24.*60., linestyle = '', 
				color = 'k', marker = 'o', ms = 4, zorder = 1)
			plt.plot(np.arange(len(model)), ttv_model*24.*60.,
				color = 'blue', linewidth = 1, zorder = 1000)

			for unc_mod in unc_models:
				unc_model = unc_mod[i]
				unc_ttv_model = unc_model - mean_ephem(
					np.arange(len(unc_model)))
				plt.plot(np.arange(len(unc_model)),
					unc_ttv_model*24.*60., color = 'blue',
					linewidth = 0.5, alpha = 0.05,
					zorder = 10)
				plt.xlim(-1, len(unc_model))
			plt.xlabel("Epoch")
			plt.ylabel("TTV [min]")
			if outname is None:
				plt.show()
				plt.close('all')
			else:
				plt.savefig(outname + f'_{i+1}.png')
				plt.close('all')
	return None

def plot_eccentricity_posteriors(system, bins = 500, outname = None):
	if system.results == None:
		raise ValueError("No retrieval found in your system object!")

	#plotting eccentricity and vectors
	samps = system.samples_equal
	thetas = [system.parse_with_fixed(samp) for samp in samps]
	plot_names = ['ecc', 'arg_peri', 'ecc_vec_h', 'ecc_vec_k']
	plot_labels = [r'$e$', r'$\omega$', r'$h = e\cos\omega$', 
		r'$k = e\sin\omega$']
	ecc = np.zeros([system.nplanets, samps.shape[0]])
	argperi = np.zeros(ecc.shape)
	h = np.zeros(ecc.shape)
	k = np.zeros(ecc.shape)
	for i in range(ecc.shape[0]):
		for j in range(ecc.shape[1]):
			_,_,hprime,kprime,_,_,_ = thetas[j][i]
			ecc[i,j] = hprime**2 + kprime**2
			argper_rad = np.arctan2(kprime, hprime)
			argperi[i,j] = argper_rad*180./np.pi
			h[i,j] = ecc[i,j]*np.cos(argper_rad)
			k[i,j] = ecc[i,j]*np.sin(argper_rad)
		for arr, name, lab in zip([ecc[i], argperi[i], h[i], k[i]], 
			plot_names, plot_labels):
			plotname = f'_{i+1}_{name}.png'
			plot_kde(arr, plotname, lab, bins, outname)

	#plotting apsidal alignments
	comb = combinations(np.arange(system.nplanets), 2)
	for comb in list(comb):
		c = sorted(comb, reverse = True)
		name = f'_apsidal_{c[0]+1}_{c[1]+1}.png'
		label = f'$\omega_{c[0]+1} - \omega_{c[1]+1}$'
		apsidal_alignment = argperi[c[0],:] - argperi[c[1],:]
		apsidal_alignment %= 360.
		apsidal_alignment -= 180.
		plot_kde(apsidal_alignment, name, label, bins, outname) 
	return None

def plot_kde(arr, name, lab, bins, outname):
	n, b = np.histogram(arr, bins = bins, density = True)
	n = norm_kde(n, 10.)
	x = 0.5 * (b[1:] + b[:-1])
	y = n
	plt.fill_between(x, y, color='b', alpha = 0.6)
	plt.xlabel(lab)
	plt.ylabel("(Integral-normalized) Probability Density")
	plt.xlim(min(x), max(x))
	plt.ylim(0, max(y)*1.05)
	if outname == None:
		plt.show()
	else:
		plt.savefig(outname + name)
		plt.close('all')
	return None

def plot_resonant_angle(system, resonance_ratio_string, sim_len_yr,
	outname = None, downsample = 1, sim_res_factor = 5):
	plt.figure(figsize = (12, 8))
	matplotlib.rcParams['font.size'] = 20
	results = system.results
	ind = np.argmax(results.logl)
	max_like_theta = results.samples[ind]

	num, denom = resonance_ratio_string.split(':')
	p = int(denom)
	q = int(num) - p #order of resonance

	times, angles = rb.track_resonant_angle(
		system, max_like_theta, p, q, sim_len_yr, sim_res_factor)
	times = times[::downsample]
	angles = angles[::downsample]

	val1 = p+q if p+q > 1 else ''
	val2 = p if p > 1 else ''
	val3 = q if q > 1 else ''
	print(len(times))
	plt.plot(times, (angles*180./np.pi) % 360, c = 'b')
	plt.xlabel('Time [yr]')
	plt.ylabel(r'${0}\lambda_2 - {1}\lambda_1 - {2}\varpi_2$'.format(
		val1, val2, val3))
	if outname == None:
		plt.show()
	else:
		plt.savefig(outname + f'_res_angle.png')
		plt.close('all')
	return None

def plot_information_timeseries(all_divs, obs_epoch = None, outname = None):
	factor_two_redux = 1-3/(8*np.log(2))
	ep, _, nplanets = all_divs.shape
	epochs = np.arange(ep)
	xmin = epochs[0]
	xmax = epochs[-1]
	for i in range(nplanets):
		cur_divs = all_divs[:,:,i]
		ymin = 1e-2
		ymax = max(max(cur_divs.flatten()), factor_two_redux)*1.2
		plt.figure(figsize = (12, 8))
		percentiles = np.percentile(cur_divs, [2.5, 50, 97.5], axis = 1)
		plt.plot(epochs, percentiles[1], 
			c = 'cornflowerblue', linestyle = '-')

		plt.fill_between(epochs, percentiles[1], 
			percentiles[0], color = 'cornflowerblue', alpha = 0.2, 
			linewidth = 0.)

		plt.fill_between(epochs, percentiles[1],
			percentiles[2], color = 'cornflowerblue', alpha = 0.2,
			linewidth = 0.)

		plt.hlines([factor_two_redux], xmin, xmax, colors = 'r',
			linestyles = '--', alpha = 0.4, 
			label = 'Approx. 2x reduction in uncertainty')

		if obs_epoch is not None:
			plt.vlines([obs_epoch], ymin, ymax, colors = 'b', 
				linestyles = '--', alpha = 0.4,
				label = 'Epoch of observation')
		plt.xlim(xmin, xmax)
		plt.ylim(ymin, ymax)
		plt.yscale('log')
		plt.xlabel("Epoch")
		plt.ylabel(f'Expected Information Gain for Planet {i+1} (bits)')
		plt.legend(loc = 'best')
		if outname == None:
			plt.show()
		else:
			plt.savefig(outname + f'_{i+1}.png')
			plt.close('all')
	return None

def plot_megno_histogram(megnos, timescale_str = '1 kyr', outname = None):
	plt.hist(megnos, bins = 100, range = (0, 10), color = 'b')
	plt.xlabel(r'MEGNO  $<Y>$ at '+timescale_str)
	plt.ylabel('Number')
	if outname == None:
		plt.show()
	else:
		plt.savefig(outname + f'_megno_histogram.png')
		plt.close('all')
	return None


def plot_ttv_data(system):
	for planet in system.planets:
		if planet.transiting:
			dat = planet.ttv
			ep = planet.epochs
			err = planet.ttv_err
			mean_ephem = planet.mean_ephem

			plt.figure(figsize = (12, 8))
			ttv_data = dat - mean_ephem(ep)
			plt.errorbar(ep, ttv_data*24.*60., yerr = err*24.*60.,
				linestyle = '', color = 'k', marker = 'o', 
				ms = 4, zorder = 1)
			plt.xlabel("Epoch")
			plt.ylabel("TTV [min]")
			plt.show()
			plt.close('all')		

def debug_plots(system, models):
	for planet, model in zip(system.planets, models):
		data = planet.ttv
		epochs = planet.epochs
		errs = planet.ttv_errs
		mean_ephem = planet.mean_ephem
		
		ttv_data = data - mean_ephem(epochs)
		ttv_model = model - mean_ephem(epochs)

		plt.plot(epochs, ttv_model)
		plt.plot(epochs, ttv_data, 'ko')
		plt.xlabel("Epoch")
		plt.ylabel("TTV [day]")
		print(data, models)
		print(mean_ephem(epochs))
		plt.show()
	return None


