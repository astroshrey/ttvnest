import numpy as np
from . import plot_utils as pu
from .constants import *
import dynesty
import ttvfast
import scipy.stats as ss
from dynesty import utils as dyfunc

class TTVSystem:
	def __init__(self, *planets, dt = None, start_time = None,
		sim_length = None, verbose = True):
		self.planets = planets
		self.nplanets = len(self.planets)
		self.start_time = start_time
		self.transiting = [pl.transiting for pl in self.planets]

		if dt is not None:
			self.dt = dt
		else:
			self.all_priors = self.get_all_priors()
			self.dt = self.calculate_dt_from_priors()

		#internal timekeeping
		self.data = np.array([p.ttv for p in self.planets \
			if p.transiting])
		alltimes = np.concatenate(self.data).ravel()
		mintime = np.floor(min(alltimes)) - buffersize
		maxtime = np.ceil(max(alltimes)) + buffersize

		if start_time is not None:
			#external timekeeping
			self.start_time = start_time
		else:
			#internal timekeeping
			self.start_time = mintime
		if sim_length is not None:
			#external timekeeping
			self.sim_length = sim_length
		else:
			#internal timekeeping
			self.sim_length = maxtime - mintime

		if verbose:
			self.print_timekeeping_standards()
		self.reference_times(self.start_time)

		self.data = np.array([p.ttv for p in self.planets \
			if p.transiting])
		self.errs = np.array([p.ttv_err for p in self.planets \
			if p.transiting])
		self.epochs = np.array([p.epochs for p in self.planets \
			if p.transiting])

		self.all_priors = self.get_all_priors()
		self.priors_for_fit = self.get_non_fixed_priors()
		self.periodic = self.get_periodic_indices()
		self.fit_param_names = self.get_fit_param_names()

		self.ndim = len(self.fit_param_names)
		self.results = None
		self.samples_equal = None
		self.max_like_result = None

		self.linear_fit_priors = self.get_linear_fit_priors()
		self.linear_fit_results = None

	def reference_times(self, reftime):
		for planet in self.planets:
			if planet.transiting:
				#move transit times back to reftime
				planet.ttv -= reftime
				
				#adjust the epochs if necessary
				new_t0 = planet.avg_t0 - reftime
				ntransits = 0
				while new_t0 < 0:
					new_t0 += planet.avg_period
					ntransits += 1
				planet.epochs -= ntransits

				#recalculate the mean ephemerides
				planet.mean_ephem = planet.get_trend(planet.ttv,
					planet.epochs, planet.ttv_err)
				planet.avg_period = planet.mean_ephem.c[0]
				planet.avg_t0 = planet.mean_ephem.c[0]

				#reset the T0 prior for transiting planets
				old_prior = list(planet.prior_dict['t0_prior'])
				old_prior[1] -= (reftime - \
					ntransits*planet.avg_period)
				if old_prior[0] == 'Uniform':
					old_prior[2] -= (reftime - \
						ntransits * planet.avg_period)
				planet.prior_dict['t0_prior'] = tuple(old_prior)

		#zero out the start time as everything is now zero ref'd
		self.start_time -= reftime 

		self.validate_epochs()
		return None 

	def validate_epochs(self):
		for i, planet in enumerate(self.planets):
			if planet.transiting:
				first_transit = planet.ttv[0] - \
					planet.epochs[0]*planet.avg_period
				if first_transit - planet.avg_period >= 0:
					raise UserWarning(
					"The epoch array is " + 
					f'incorrect for planet {i+1}: first '+
					"transit was calculated to be " + 
					f'{first_transit}, but this is ' +
					'at least the second transit')		

	def print_timekeeping_standards(self):
		print("Simulation start/reference time: ", self.start_time)
		print("ttvnest timestep: ", self.dt)
		print("Simulation length: ", self.sim_length)
		print("Simulation end time: ",
			self.start_time + self.sim_length)
		return None
	
	def calculate_dt_from_priors(self):
		periods = []
		for i in range(self.nplanets):
			prior = self.all_priors[i*7+1]
			periods.append(prior[2])
		min_timescale = min(periods)
		#to avoid timestep-induced chaos you want dt < min(P)/20
		return min_timescale / 25 

	def get_all_priors(self):
		all_priors = []
		for i, planet in enumerate(self.planets):
			pl = planet.prior_dict.values()
			all_priors += [(i,) + tup for tup in pl]
		return all_priors

	def get_non_fixed_priors(self):
		ap = self.all_priors
		return [tup for tup in ap if tup[1] != 'Fixed']

	def get_periodic_indices(self):
		periodic = []
		for i, prior in enumerate(self.priors_for_fit):
			if prior[1] == 'Periodic':
				periodic.append(i)
		if len(periodic) < 1:
			return None
		return periodic

	def prior_transform(self, u):
		def unif(u, lo, up):
			return lo + u*(up - lo)
	
		def per(u, lo, up):
			return lo + (u % 1.)*(up - lo)

		transform_dict = {'Uniform': unif,
				'Normal': ss.norm.ppf,
				'Periodic': per}

		x = np.array(u) #copy unit cube
		for i, prior in enumerate(self.priors_for_fit):
			f_i = transform_dict[prior[1]]
			x[i] = f_i(u[i], *prior[2:])

		return x	

	def get_fit_param_names(self):
		priors = self.priors_for_fit
		fit_param_names = []
		for j, prior in enumerate(self.all_priors):
			if prior[1] != 'Fixed':
				i = str(prior[0] + 1)
				if j % 7 == 6 and \
				not self.planets[prior[0]].transiting:
					fit_param_names.append(
			r'$\mathcal{M}_'+i+'\ [\mathrm{degrees}]$')
				else:
					conversion_arr = [
				r'$M_'+i+r'/M_\star/3\times10^{-6}$',
				r'$P_'+i+'\ [\mathrm{days}]$',
				r'$\sqrt{e}_'+i+'\cos(\omega_'+i+')$',
				r'$\sqrt{e}_'+i+'\sin(\omega_'+i+')$', 
				r'$i_'+i+'\ [\mathrm{degrees}]$',
				r'$\Omega_'+i+'\ [\mathrm{degrees}]$',
				r'$T_{0,'+i+'}\ [\mathrm{days}]$']
					fit_param_names.append(
						conversion_arr[j % 7])
		return fit_param_names

	def parse_with_fixed(self, theta):
		theta = np.append(theta, [0]*(self.nplanets*7 - len(theta)))
		for i, prior in enumerate(self.all_priors):
			if prior[1] == 'Fixed':
				theta = theta[:-1]
				theta = np.insert(theta, i, prior[2])
		thetas = np.split(theta, self.nplanets)
		return thetas

	def t0_to_mean_anom(self, t0, e, w_deg, P, t_epoch):
		w = w_deg * np.pi / 180. #omega in radians
		f_0 = np.pi/2 - w       #true anomaly at inf conj
		E_0 = 2*np.arctan(np.tan(f_0/2) * np.sqrt((1-e)/(1+e)))
		M_0 = E_0 - e*np.sin(E_0)
		#above are eccentric and mean anomalies at inf conj

		t_peri = t0 - P*M_0/(2*np.pi)
		mean_anomaly = 360./P*(t_epoch - t_peri)
		return mean_anomaly % 360

	def forward_model(self, theta):
		start = self.start_time
		end = start + self.sim_length
		system = []
		thetas = self.parse_with_fixed(theta)
		for i_pl, arr in enumerate(thetas):
			if self.planets[i_pl].transiting:
				mp_ms, P, hprime, kprime, i, Omega, t0 = arr
				e = hprime**2 + kprime**2
				if e > e_cutoff:
					return None
				w = np.arctan2(kprime, hprime)*180./np.pi
				t0 += start
				M = self.t0_to_mean_anom(t0, e, w, P, start)
			else:
				mp_ms, P, hprime, kprime, i, Omega, M = arr	

			e = hprime**2 + kprime**2
			if e > e_cutoff:
				return None
			w = np.arctan2(kprime, hprime)*180./np.pi
			planet_params = [mp_ms*earth_per_sun, P, e, i, 
				Omega, w, M]
			system.append(ttvfast.models.Planet(*planet_params))
			
		results = ttvfast.ttvfast(system, stellarmass, start, 
			self.dt, end)
		results = results['positions']

		planet = np.array(results[0], dtype = 'int')
		time = np.array(results[2], dtype = 'float')
		timing_arrays = []
		for i, transiting in enumerate(self.transiting):
			if transiting:
				timing_arr = time[planet == i]
				timing_arr = timing_arr[timing_arr != -2.0]
				timing_arr -= start
				timing_arrays.append(timing_arr)
		return np.array(timing_arrays)

	def log_likelihood(self, theta, data, errs, epochs, debug = False):
		# calculate the model
		models = self.forward_model(theta)
		if models is not None:
			try:	
				#then match model to observed epochs
				#(this accounts for observation gaps
				models = np.array([model[epoch] for \
					model, epoch in zip(models, epochs)])
				if debug:
					print(theta)
					pu.debug_plots(self, models)
			except IndexError as e:
			#happens when ttvfast returns fewer transit events
			#over the integration domain than the data contain
				return -1e300
		else:
			return -1e300	
		#then calculate log likelihood sum
		log_like = 0
		for model, datum, err in zip(models, data, errs):
			residsq = (model - datum)**2 / err**2
			log_like -= 0.5 * np.sum(residsq + np.log(
						2*np.pi*err**2))
		
		if not np.isfinite(log_like):
			return -1e300
		return log_like

	def posterior_summary(self, filename = None, equal_samples = False,
		ndig = 6):

		results = self.results
		samples = results.samples
		nplanets = self.nplanets
		labels = self.fit_param_names
		if not equal_samples:
			weights = np.exp(results.logwt - results.logz[-1])
			quants = [dyfunc.quantile(samples[:,i],
				[0.025, 0.5, 0.975], weights = weights) for \
				i in range(samples.shape[1])]
		else:
			quants = [dyfunc.quantile(samples[:,i],
				[0.025,0.5,0.975]) for i in range(
				samples.shape[1])]

		if filename is not None:
			f = open(filename, 'w')
		else:
			f = None
		
		print("Summary (middle 95 percentile): ", file = f)
		for quant, label in zip(quants, labels):
			qlo, qmid, qhi = quant
			qup = qhi - qmid
			qdown = qmid - qlo
			print(label + ': $' + str(round(qmid,
				ndig)) + '^{+' + str(round(qup, ndig)) + 
				'}_{-' + str(round(qdown, ndig)) + '}$',
				file = f)
		if f is not None:
			f.close()
		return filename

	def retrieve(self, retriever = 'DynamicNestedSampler', dlogz = 0.01, 
		nlive = 1000, wt_kwargs = None,
		**dynesty_kwargs):
		if retriever == 'NestedSampler':
			fn = dynesty.NestedSampler
			dynesty_kwargs['nlive'] = nlive
			dynesty_kwargs['dlogz'] = dlogz
		elif retriever == 'DynamicNestedSampler':
			fn = dynesty.DynamicNestedSampler
			dynesty_kwargs['wt_kwargs'] = wt_kwargs
			dynesty_kwargs['nlive_init'] = nlive
			dynesty_kwargs['dlogz_init'] = dlogz
		else:
			raise ValueError("Unrecognized Sampler!")
		samp_kw = fn.__code__.co_varnames
		filter_by_key = lambda keys: {x: dynesty_kwargs[x] for \
			x in keys if x in dynesty_kwargs}
		samp_kwargs = filter_by_key(samp_kw)
			
		sampler = fn(self.log_likelihood, self.prior_transform,
			self.ndim, logl_args = (self.data, 
			self.errs, self.epochs), periodic = self.periodic, 
			**samp_kwargs)

		run_kw = sampler.run_nested.__code__.co_varnames
		run_kwargs = filter_by_key(run_kw)
		sampler.run_nested(**run_kwargs)

		results = sampler.results
		self.results = results

		samples = results.samples
		weights = np.exp(results.logwt - results.logz[-1])
		samples_equal = dyfunc.resample_equal(samples, weights)
		self.samples_equal = samples_equal
		
		ind = np.argmax(results.logl)
		max_like_result = samples[ind]
		self.max_like_result = max_like_result

		return self.results


############## linear model below ################################

	def linear_forward_model(self, theta):
		thetas = np.split(theta, self.nplanets)
		timing_arrs = []
		for t, ep in zip(thetas, self.epochs):
			m, b = t
			timing_arrs.append(m*ep + b)
		return np.array(timing_arrs)

	def linear_model_log_like(self, theta, data, errs, epochs):
		# calculate the model
		models = self.linear_forward_model(theta)
		
		#then calculate log likelihood sum
		log_like = 0
		for model, datum, err in zip(models, data, errs):
			residsq = (model - datum)**2 / err**2
			log_like -= 0.5 * np.sum(residsq + np.log(
				2*np.pi*err**2))
	
		if not np.isfinite(log_like):
			return -1e300
		return log_like

	def get_linear_fit_priors(self):
		all_priors = []
		for i, planet in enumerate(self.planets):
			if planet.transiting:
				pl = planet.prior_dict
				all_priors += [(i,) + pl['period_prior'],
					 (i,) + pl['t0_prior']]
		return all_priors

	def linear_ptform(self, u):
		def unif(u, lo, up):
			return lo + u*(up - lo)
	
		def per(u, lo, up):
			return lo + (u % 1.)*(up - lo)

		transform_dict = {'Uniform': unif,
				'Normal': ss.norm.ppf,
				'Periodic': per}

		x = np.array(u) #copy unit cube
		for i, prior in enumerate(self.linear_fit_priors):
			f_i = transform_dict[prior[1]]
			x[i] = f_i(u[i], *prior[2:])

		return x	

	def linear_model_retrieve(self, **dynesty_kwargs):
		fn = dynesty.NestedSampler
		samp_kw = fn.__code__.co_varnames
		filter_by_key = lambda keys: {x: dynesty_kwargs[x] for \
			x in keys if x in dynesty_kwargs}
		samp_kwargs = filter_by_key(samp_kw)
		ntransiting = np.sum(self.transiting)
		ndim = ntransiting*2
		
		sampler = fn(self.linear_model_log_like, self.linear_ptform,
			ndim, logl_args = (self.data, 
			self.errs, self.epochs), **samp_kwargs)

		run_kw = sampler.run_nested.__code__.co_varnames
		run_kwargs = filter_by_key(run_kw)
		sampler.run_nested(**run_kwargs)

		self.linear_fit_results = sampler.results
		return self.linear_fit_results



