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
