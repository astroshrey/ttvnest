import numpy as np
import matplotlib.pyplot as plt
from . import forward_model as fm
from . import retrieval as ret
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from .constants import *
import matplotlib
import random
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20

def gen_labels(nplanets):
	labels = []
	for i in range(nplanets):
		num = str(i + 1)
		labels += [r'$M_'+num+'/M_\star$',
			r'$P_'+num+'\ [\mathrm{days}]$', 
			r'$e_'+num+'\cos(\omega_'+num+')$', 
			r'$e_'+num+'\sin(\omega_'+num+')$',
			r'$\mathcal{M}_'+num+'\ [^\circ]$']
	return labels	

def dynesty_plots(dresults, nplanets):
	labels = gen_labels(nplanets)
	ndim = len(labels)

	plt.figure(figsize = (20, 20))
	cfig, caxes = dyplot.cornerplot(dresults, color = 'blue',
		max_n_ticks = 3, labels = labels)
	plt.show()

	plt.figure(figsize = (20, 20))
	rfig, raxes = dyplot.runplot(dresults)
	plt.show()
	
	plt.figure(figsize = (20, 20))
	tfig, taxes = dyplot.traceplot(dresults, truths = np.zeros(ndim), 
		show_titles = True, trace_cmap = 'viridis', labels = labels)
	plt.show()

	return None

def plot_results(dresults, data, errs, uncertainty_curves = 0, dt = 0.1,
	sim_length = 2000):

	samples = dresults.samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])
	median_result = [dyfunc.quantile(samples[:,i], [0.5], 
		weights = weights)[0] for i in range(samples.shape[1])]
	samples_equal = dyfunc.resample_equal(samples, weights)
	inds = np.arange(samples_equal.shape[0])
	pick = np.random.choice(inds, uncertainty_curves, replace=False)
	random_samples = samples_equal[pick,:]

	#data and median result
	n_planets = int(samples.shape[1]/5)
	paramv = [median_result[i*5:(i+1)*5] for i in range(n_planets)]
	models = fm.run_simulation(1., dt, sim_length, *paramv)
	epochs = [ret.get_inds(model, dat) for model, dat in zip(models, data)]

	#uncertainty results
	unc_models = []
	unc_epochs = []
	for samp in random_samples:
		rand_paramv = [samp[i*5:(i+1)*5] for i in range(n_planets)]
		rand_models = fm.run_simulation(1., dt, sim_length,
			*rand_paramv)
		rand_epochs = [ret.get_inds(model, dat) for model, dat in zip(
			rand_models, data)]
		unc_epochs.append(rand_epochs)
		unc_models.append(rand_models)

	for i in range(n_planets):
		datum = data[i]
		model = models[i]
		err = errs[i]
		epoch = epochs[i]
		datum_trend = get_trend(datum, epoch)
		model_trend = get_trend(model[epoch], epoch)
		ttv_datum = datum - datum_trend(epoch)
		ttv_model = model - model_trend(np.arange(len(model)))
		plt.errorbar(datum, ttv_datum*24.*60., yerr = err*24.*60., 
			linestyle = '', color = 'k', marker = 'o', ms = 4, 
			zorder = 1)
		plt.plot(model, ttv_model*24.*60., color = 'blue', 
			linewidth = 1, zorder = 1000)
		for unc_mod, unc_ep in zip(unc_models, unc_epochs):
			unc_model = unc_mod[i]
			unc_epoch = unc_ep[i]
			unc_model_trend = get_trend(unc_model[unc_epoch],
				unc_epoch)
			unc_ttv_model = unc_model - unc_model_trend(
				np.arange(len(unc_model)))
			plt.plot(unc_model, unc_ttv_model*24.*60., 
				color = 'blue', linewidth = 0.5, alpha = 0.05,
				zorder = 10)

		plt.xlabel("Time [BJD - 2454900]")
		plt.ylabel("TTV [min]")
		plt.show()
	return None

def get_trend(data, obsind):
	z = np.polyfit(obsind, data, 1)
	p = np.poly1d(z)
	return p

def plot_results_with_uncertainty(theta, data, errs, dt = 0.1, sim_length = 2000):
	n_planets = int((len(theta))/5)
	paramv = [theta[i*5:(i+1)*5] for i in range(n_planets)]
	models = fm.run_simulation(1., dt, sim_length, *paramv)
	epochs = [ret.get_inds(model, dat) for model, dat in zip(models, data)]
	models = np.array([model[epoch] for model, epoch in \
		zip(models, epochs)])
	plot_ttv(data, models, errs, epochs, paramv)
	plot_resid(data, models, errs, epochs, paramv)
	return None

def plot_ttv_from_params(theta, data, errs, dt = 0.1, sim_length = 2000):
	n_planets = int((len(theta))/5)
	paramv = [theta[i*5:(i+1)*5] for i in range(n_planets)]
	models = fm.run_simulation(1., dt, sim_length, *paramv)
	epochs = [ret.get_inds(model, dat) for model, dat in zip(models, data)]
	models = np.array([model[epoch] for model, epoch in \
		zip(models, epochs)])
	plot_ttv(data, models, errs, epochs, paramv)
	plot_resid(data, models, errs, epochs, paramv)
	return None

def plot_ttv(data, models, errs, epochs, paramv):
	for datum, model, err, epoch, params in zip(data, models, errs, epochs, paramv):    
		ttv_datum = detrend(datum, epoch)
		ttv_model = detrend(model, epoch)
		plt.errorbar(datum, ttv_datum*24.*60., yerr = err*24.*60., linestyle = '',
			color = 'k', marker = 'o')
		plt.plot(model, ttv_model*24.*60., ms = 2, color = 'blue', linewidth = 1)
		plt.xlabel("Time [BJD - 2454900]")
		plt.ylabel("TTV [min]")
		plt.show()
	return None

def plot_resid(data, models, errs, epochs, paramv):
	for datum, model, err, epoch, params in zip(data, models, errs, epochs, paramv):    
		resid = datum - model
		plt.errorbar(epoch, resid*24.*60., yerr = err*24*60., linestyle = '', color = 'r', marker= 'o')
		plt.xlabel("Epoch")
		plt.ylabel("Residual from TTV model [min]")
		plt.show()
	return None

def detrend(data, obsind):
	z = np.polyfit(obsind, data, 1)
	p = np.poly1d(z)
	return data - p(obsind)
