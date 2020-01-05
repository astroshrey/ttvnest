import numpy as np
from . import forward_model as fm
from . import plot_utils as pu
from .constants import *
import dynesty
from dynesty import utils as dyfunc

def retrieve(nplanets, prior_transform, fixed_params, data, errs, 
	sampler = 'rwalk', bounder = 'multi', pool = None, queue_size = None, 
	dt = 0.1, sim_length = 2000, start_time = tkep, init_live = None, 
	dlogz_init = 0.1, transiting = None, n_effective = None,
	maxiter = None, maxcall = None):
	ndim = nplanets*7 - len(fixed_params)
	if init_live is None:
		init_live = 100*ndim
	if transiting is None:
		transiting = np.arange(nplanets)
	periodic = get_periodic_indices(nplanets, fixed_params)

	#the mean anomaly is always periodic, so do this for every planet
#	periodic = [5*i + 4 for i in range(nplanets)]


	print("Running dynesty with the {0} sampler, {1} bounding, and {2} initial live points...".format(
		sampler, bounder, init_live))
	dsampler = dynesty.NestedSampler(log_likelihood, 
		prior_transform, ndim, logl_args = (data, errs, dt, start_time,
		sim_length, fixed_params, nplanets, transiting),
		sample = sampler, bound = bounder, periodic = periodic,
		pool = pool, queue_size = queue_size, nlive = init_live)

	dsampler.run_nested(dlogz = dlogz_init)
	dresults = dsampler.results

	return dresults

def posterior_summary(results, filename = None, equal_samples = False):
	samples = results.samples
	if not equal_samples:
		weights = np.exp(results.logwt - results.logz[-1])
		quants = [dyfunc.quantile(samples[:,i], [0.025, 0.5, 0.975],
			weights = weights) for i in range(samples.shape[1])]
	else:
		quants = [dyfunc.quantile(samples[:,i],[0.025,0.5,0.975]) \
			for i in range(samples.shape[1])]
	nplanets = int(samples.shape[1]/5)
	labels = pu.gen_labels(nplanets)
	
	if filename is not None:
		f = open(filename, 'w')
	else: 
		f = None
	print("Summary: ", file = f)
	for quant, label in zip(quants, labels):
		qlo, qmid, qhi = quant
		qup = qhi - qmid
		qdown = qmid - qlo
		print(label + ': $' + str(qmid) + '^{+' + str(qup) + '}_{-' + 
			str(qdown) + '}$', file = f)
	if f is not None:
		f.close()	
	return filename

def log_likelihood(theta, data, errs, dt, start_time, sim_length, fixed_params,
	nplanets, transiting):
	#first check that the dimensions are correct
	try:
		#then unpack the parameters
		stellarmass = 1.
		#paramv = [theta[i*5: (i + 1)*5] for i in range(nplanets)]

		#then calculate the model
		models = fm.run_simulation(stellarmass, dt, start_time,
				sim_length, theta, fixed_params, nplanets,
				transiting)

		#then discover the epoch arrays
		epochs = np.array([get_inds(model, datum) for model, datum in \
			zip(models, data)])
	
		#then match model to observed epochs 
		#(this accounts for observation gaps)
		models = np.array([model[epoch] for model, epoch in \
			zip(models, epochs)])
		"""
		import matplotlib.pyplot as plt
		datum_trend = pu.get_trend(data[0], epochs[0], errs[0])
		ttv_model = models[0] - datum_trend(epochs[0])
		plt.plot(epochs[0], ttv_model)
		plt.plot(epochs[0], data[0] - datum_trend(epochs[0]), 'ko')
		plt.xlabel("Epoch")
		plt.ylabel("TTV [day]")
		print(data[0], models[0])
		print(datum_trend(epochs[0]))
		print(theta)
		plt.show()
		"""
	except ValueError as e:
		return -1e300

	#then calculate log likelihood sum
	log_like = 0
	for model, datum, err in zip(models, data, errs):
		residsq = (model - datum)**2 / err**2
		log_like -= 0.5 * np.sum(residsq + np.log(2*np.pi*err**2))

	if not np.isfinite(log_like):
		return -1e300
	
	return log_like

def find_nearest(array, value):
	return (np.abs(array - value)).argmin()

def get_inds(modeled, observed):
	idxarr = [find_nearest(modeled, val) for val in observed]
	return idxarr

def get_periodic_indices(nplanets, fixed_params):
	#LongNode and MeanAnom are always periodic
	allinds = np.zeros(7*nplanets)
	for i in range(nplanets):
		allinds[7*i + 5] = 1 #LongNode
#		allinds[7*i + 6] = 1 #MeanAnom
	delinds = []
	for fixed in fixed_params:
		delinds.append(fixed[0]*7+fm.param_id(fixed[1]))
	allinds = np.delete(allinds, delinds)
	return np.argwhere(allinds == 1).flatten()
