import numpy as np
from dynesty import utils as dyfunc
from scipy.ndimage import gaussian_filter as norm_kde
from . import forward_model as fm
import matplotlib.pyplot as plt
import scipy

def calculate_information_timeseries(results, measurement_uncertainty,
	measured_planet, stellarmass = 1, dt = 0.1, sim_length = 7305., 
	nsamps = 100):

	samples = results.samples
	nplanets = int(samples.shape[1]/5)
	weights = np.exp(results.logwt - results.logz[-1])

	#first resmple the posterior to be equal
	samples_equal = dyfunc.resample_equal(samples, weights)
	#then need to propogate all models forward
	print("Propogating all models in posterior forward to time {0}...".format(sim_length))
	models_all = propogate_all_models(samples_equal, nplanets, 
		stellarmass, dt, sim_length)
	#finally calculate the distribution of information gains at each time
	print("Calculating the Kullback-Leibler Divergence distribution at each epoch...")
	all_divs = calculate_dkl_timeseries(models_all, samples_equal,
		measured_planet, measurement_uncertainty, nsamps)
	return all_divs

def propogate_all_models(samples_equal, nplanets, stellarmass, dt, sim_length):
	models_all = []
	n_samples = samples_equal.shape[0]
	for i in range(n_samples):
		theta = samples_equal[i,:]
		paramv = [theta[i*5: (i + 1)*5] for i in range(nplanets)]
		models = fm.run_simulation(stellarmass, dt, sim_length, *paramv)
		models_all.append(models)
		perc = int(n_samples/100)
		if i % perc == 0:
			print(str(i/perc)+'% complete')
	return models_all


def calculate_dkl_timeseries(models_all, samples_equal, measured_planet, 
	measurement_uncertainty, nsamps):
	model_len = len(models_all[0][measured_planet])
	all_divs = np.zeros([model_len, nsamps])
	#for each time, calculate and save the dkl distribution
	for i in range(model_len):
		all_divs[i,:] = get_dkl_distribution(i, models_all, 
			samples_equal, measured_planet, measurement_uncertainty,
			nsamps)
		#progress bar
		perc = int(model_len/100)
		if i % perc == 0:
			print(str(i/perc)+'% complete')
	return all_divs

def get_dkl_distribution(epoch, models_all, samples_equal, measured_planet,
	measurement_uncertainty, nsamps):
	dkls = np.zeros(nsamps)
	for j in range(nsamps):
		#pick random sample to be "true"
		true_ind = np.random.randint(0, len(models_all))
		true_val = models_all[true_ind][measured_planet][epoch]
	
		#get probabilities for all models given the "true" random sample
		test_times = np.array([models_all[k][measured_planet][epoch] \
			for k in range(len(models_all))])
		probs = gp(test_times, true_val, measurement_uncertainty)
	
		#sample the models based on the probabilities above
		rands = np.random.random(len(models_all))
		new_samples_equal = samples_equal[probs > rands]

		#finally calculate the Kullback-Leibler divergence on 
		#KDEs of the old and new distributions
		dkls[j] = D_KL(*KDEs(samples_equal[:,0],
			new_samples_equal[:,0], plot = False))
		#TODO: CHANGE IND 0 ABOVE TO WHATEVER INDEX OR MERIT USER WANTS
	return dkls

def get_prob(test, true, unc):
	return 1 - scipy.stats.chi2._cdf((test - true)**2/(unc**2), 1)
gp = np.vectorize(get_prob) 
#vectorizing makes it much faster to compute chi2 probabilities for an array

def KDEs(x_old, x_new, bins = 500, plot = False, color1 = 'gray',
	color2 = 'dodgerblue'):

	#set the bins and ranges based on the pre-rejection sampled data
	span = 0.999999426697
	q = [0.5 - 0.5 * span, 0.5 + 0.5 * span]
	ranges = dyfunc.quantile(x_old, q)

	#KDE of pre-rejection sampled data
	n1, b1 = np.histogram(x_old, bins = bins, range = ranges, 
		density = True)
	n1 = norm_kde(n1, 10.)
	x1 = 0.5 * (b1[1:] + b1[:-1])
	y1 = n1
	if plot:
		plt.fill_between(x1, y1, color=color1, alpha = 0.5)
	
	#KDE of post-rejection sampled data
	n2, b2 = np.histogram(x_new, bins = bins, range = ranges, 
		density = True)
	n2 = norm_kde(n2, 10.)
	x2 = 0.5 * (b2[1:] + b2[:-1])
	y2 = n2
	if plot:
		plt.fill_between(x2, y2, color=color2, alpha = 0.5)
		plt.show()
	return n1, n2

def D_KL(n1, n2):
	mask = np.where(n2 != 0.)
	return np.nansum(n1[mask]*np.log2(n1[mask]/n2[mask]))


