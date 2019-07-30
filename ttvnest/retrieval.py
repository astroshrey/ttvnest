import numpy as np
from . import forward_model as fm
from . import plot_utils as pu
import dynesty
from dynesty import utils as dyfunc

def retrieve(nplanets, prior_transform, data, errs):
	ndim = nplanets*5

	#the mean anomaly is always periodic, so do this for every planet
	periodic = [5*i + 4 for i in range(nplanets)]

	dsampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform,
		ndim, logl_args = (data, errs), sample = 'rwalk', 
		bound = 'multi')
	dsampler.run_nested(wt_kwargs = {'pfrac':1.0})

	dresults = dsampler.results
	return dresults

def posterior_summary(results, filename = None):
	samples = results.samples
	weights = np.exp(results.logwt - results.logz[-1])
	samples_equal = dyfunc.resample_equal(samples, weights)
	quants = [dyfunc.quantile(samples[:,i], [0.025, 0.5, 0.975],
		weights = weights) for i in range(samples.shape[1])]

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

def log_likelihood(theta, data, errs, dt = 0.1, sim_length = 2000):
	#first check that the dimensions are correct
	nplanets = int(len(theta)/5)
	if not (nplanets == len(data) == len(errs)):
		err = "The number of planets is different than the number of \
			datasets!"
		raise ValueError(err)

	try:
		#then unpack the parameters
		stellarmass = 1.
		paramv = [theta[i*5: (i + 1)*5] for i in range(nplanets)]

		#then calculate the model
		models = fm.run_simulation(stellarmass, dt, sim_length, *paramv)

		#then discover the epoch arrays
		epochs = np.array([get_inds(model, datum) for model, datum in \
			zip(models, data)])
	
		#then match model to observed epochs 
		#(this accounts for observation gaps)
		models = np.array([model[epoch] for model, epoch in \
			zip(models, epochs)])
	except ValueError as e:
		return -1e300

	#then calculate log likelihood sum
	log_likelihood = 0
	for model, datum, err in zip(models, data, errs):
		residsq = (model - datum)**2 / err**2
		log_likelihood -= 0.5 * np.sum(residsq + np.log(2*np.pi*err**2))

	if not np.isfinite(log_likelihood):
		return -1e300
	
	return log_likelihood

def find_nearest(array, value):
	return (np.abs(array - value)).argmin()

def get_inds(modeled, observed):
	idxarr = [find_nearest(modeled, val) for val in observed]
	return idxarr
