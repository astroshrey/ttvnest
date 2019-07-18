import numpy as np
import matplotlib.pyplot as plt
import forward_model as fm
import retrieval as ret
from constants import *
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20

def plot_ttv_from_result(theta, data, errs, dt = 0.1, sim_length = 2000):
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
			color = 'g', marker = 'o')
		plt.plot(model, ttv_model*24.*60., 'ro', ms = 2, color = 'k')
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
