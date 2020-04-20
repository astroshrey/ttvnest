import numpy as np
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from scipy.ndimage import gaussian_filter as norm_kde
from .constants import *
import matplotlib
import random
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20

def dynesty_plots(system, truthvals = None, outname = None):
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
	plt.figure(figsize = (20, 20))
	rfig, raxes = dyplot.runplot(system.results)
	if outname == None:
		plt.show()
	else:
		plt.savefig(names[2])
		plt.close('all')
		return names
	return None
		


def plot_results(system, uncertainty_curves = 0, sim_length = 2000,
	outname = None):
	
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

	#uncertainty results
	unc_models = []
	for samp in random_samples:
		rand_models = system.forward_model(samp)
		unc_models.append(rand_models)

	for i, transiting in enumerate(system.transiting):
		if transiting:
			datum = system.data[i]
			err = system.errs[i]
			epoch = system.epochs[i]
			model = models[i]

			datum_trend = get_trend(datum, epoch, err)
			ttv_datum = datum - datum_trend(epoch)
			ttv_model = model - datum_trend(np.arange(len(model)))

			plt.figure(figsize = (12, 8))
			plt.errorbar(epoch, ttv_datum*24.*60., 
				yerr = err*24.*60., linestyle = '', 
				color = 'k', marker = 'o', ms = 4, zorder = 1)
			plt.plot(np.arange(len(model)), ttv_model*24.*60.,
				color = 'blue', linewidth = 1, zorder = 1000)

			for unc_mod in unc_models:
				unc_model = unc_mod[i]
				unc_ttv_model = unc_model - datum_trend(
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
				plt.savefig(outname + f'_{i}.png')
				plt.close('all')
	return None

def get_trend(data, obsind, errs, get_uncertainty = False):
	z = np.polyfit(obsind, data, 1, w = 1/errs)
	p = np.poly1d(z)
	return p

def plot_apsidal_alignment(results, nplanets = 2, bins = 500):
	if nplanets != 2:
		print("Only supported for two-planet systems right now")
		return None
	samples = results.samples
	weights = np.exp(results.logwt - results.logz[-1])
	samples_equal = dyfunc.resample_equal(samples, weights)
	ecosw_ind = 2
	esinw_ind = 3
	planet_1_ecosw = samples_equal[:,ecosw_ind]
	planet_1_esinw = samples_equal[:,esinw_ind]
	planet_2_ecosw = samples_equal[:,ecosw_ind + 5]
	planet_2_esinw = samples_equal[:,esinw_ind + 5]
	planet_1_w = np.arctan2(planet_1_esinw, planet_1_ecosw)*180./np.pi
	planet_2_w = np.arctan2(planet_2_esinw, planet_2_ecosw)*180./np.pi
	apsidal_alignment = planet_2_w - planet_1_w
	
	span = 0.999999426697
	q = [0.5 - 0.5 * span, 0.5 + 0.5 * span]
	ranges = dyfunc.quantile(apsidal_alignment, q)
	n1, b1 = np.histogram(apsidal_alignment, bins = bins, range = ranges,
		density = True)
	n1 = norm_kde(n1, 10.)
	x1 = 0.5 * (b1[1:] + b1[:-1])
	y1 = n1
	plt.fill_between(x1, y1, color='b', alpha = 0.5)
	plt.xlabel(r'$\omega_2 - \omega_1$ [$\degree$]')
	plt.show()
	return None

def plot_information_timeseries(all_divs, obs_epoch = None):
	plt.figure(figsize = (12, 8))
	percentiles = np.percentile(all_divs, [2.5, 50, 97.5], axis = 1)
	plt.plot(np.arange(all_divs.shape[0]), percentiles[1], 
		c = 'cornflowerblue', linestyle = '-')

	plt.fill_between(np.arange(all_divs.shape[0]), percentiles[1], 
		percentiles[0], color = 'cornflowerblue', alpha = 0.2, 
		linewidth = 0.)

	plt.fill_between(np.arange(all_divs.shape[0]), percentiles[1],
		percentiles[2], color = 'cornflowerblue', alpha = 0.2,
		linewidth = 0.)

	plt.hlines([1-3/(8*np.log(2))], 0, all_divs.shape[0], colors = 'r',
		linestyles = '--', alpha = 0.4, 
		label = 'Approx. 2x reduction in uncertainty')

	if obs_epoch is not None:
		plt.vlines([obs_epoch], 0.01, max(all_divs.flatten())*1.2,
			colors = 'b', linestyles = '--', alpha = 0.4,
			label = 'Epoch of observation')

	plt.ylim(0.01, max(all_divs.flatten())*1.2)
	plt.xlim(0, all_divs.shape[0])
	plt.yscale('log')
	plt.xlabel("Epoch")
	plt.ylabel("Expected Information Gain (bits)")
	plt.legend(loc = 'best')
	plt.show()
	return None

def debug_plots(nplanets, data, epochs, errs, models):
	for i in range(nplanets):
		datum_trend = get_trend(data[i], epochs[i], errs[i])
		ttv_model = models[i] - datum_trend(epochs[i])
		plt.plot(epochs[i], ttv_model)
		plt.plot(epochs[i], data[i] - datum_trend(epochs[i]), 'ko')
		plt.xlabel("Epoch")
		plt.ylabel("TTV [day]")
		print(data[i], models[i])
		print(datum_trend(epochs[i]))
		plt.show()
	return None
