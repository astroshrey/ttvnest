import dynesty
import numpy as np
from dynesty import utils as dyfunc

class TTVPlanet:
	"""
	A planet to include in the TTV model. Optionally initialized
	with timing data (if the planet is transiting) and priors
	on the dynamical parameters.

	All priors are specified with a tuple of shape (2,) or (3,) where
	the first elements denotes the type of prior (the prior transform
	for nested sampling is done internally). Currently, four priors are
	supported with the following first elements:
	'Fixed', with the second element being the fixed value
	'Normal', with the second and third elements specifying `N(a, b)`.
	'Uniform', with the second and third elements specifying `U(a, b)`. 
	'Periodic', with the second and third elements specifying `U(a, b)` and
	periodic boundary conditions (useful when uniformly samping angles).

	Parameters
	----------
	ttv : `~numpy.ndarray` with shape `(len_ttv,)`, optional
		Timing data for the planet if it is transiting.
	ttv_err : `~numpy.ndarray` with shape `(len_ttv,)`, optional
		Timing errors for the planet if it is transiting.
	epochs : `~numpy.ndarray` with shape `(len_ttv,)`, optional
		Epochs for the transit times in `ttv`.
	mass_prior : tuple with shape (2,) or (3,)
		The prior on planetary dynamical mass, `M_p/M_star`, in
		units of `M_Earth/M_Sun = 3e-6`. Default is `U(0, 100)`. 
	period_prior : tuple with shape (2,) or (3,)
		The prior on orbital period, in units of days. Default is 
		`U(1, 100)`. 
	h_prime_prior : tuple with shape (2,) or (3,)
		The prior on `sqrt(e)cos(omega)`. Default is `U(-1, 1)`. 
	k_prime_prior : tuple with shape (2,) or (3,)
		The prior on `sqrt(e)sin(omega)`. Default is `U(-1, 1)`. 
	inc_prior : tuple with shape (2,) or (3,)
		The prior on the orbital inclination in degrees. Default is
		fixed to 90.
	longnode_prior : tuple with shape (2,) or (3,)
		The prior on the longitude of ascending node in degrees.
		Default is fixed to 0.
	t0_prior : tuple with shape (2,) or (3,)
		The prior on the time of first transit in days, referenced
		to the system provided in `ttv`. If the planet is transiting
		(`ttv`, `ttv_err`, and `epochs` are not None), this will be
		set by default to `U(t0_avg - 100*sigma, t0_avg + 100*sigma)`,
		where t0_avg is computed from a linear fit to the data and
		sigma is the first error bar in `ttv_err`. Otherwise, if the
		planet is not transiting, this defaults to None and is unused.
	meananom_prior : tuple with shape (2,) or (3,)
		The prior on the mean anomaly in degrees. Will be used if
		the planet is not transiting. Default is periodic on [0, 360].

	"""
	
	def __init__(self, ttv = None, ttv_err = None, epochs = None,
		mass_prior = ('Uniform', 0., 100.),
		period_prior = ('Uniform', 1., 100.),
		h_prime_prior = ('Uniform', -1., 1.), 
		k_prime_prior = ('Uniform', -1., 1.),
		inc_prior = ('Fixed', 90.),
		longnode_prior = ('Fixed', 0.),
		t0_prior = None,
		meananom_prior = ('Periodic', 0., 360.),
		n_per = None):

		self.transiting = self.validate_input(ttv, ttv_err,
			epochs)
		if self.transiting:
			self.ttv = np.array(ttv)
			self.ttv_err = np.array(ttv_err)
			self.epochs = np.array(epochs)
			try:
				self.mean_ephem = self.get_trend(self.ttv, 
					self.epochs, self.ttv_err)
				self.avg_period = self.mean_ephem.c[0]
				self.avg_t0 = self.mean_ephem.c[1]
			except np.linalg.LinAlgError as e:
				print("Can't get mean ephemerides")
				self.mean_ephem = None
				self.avg_period = None
				self.avg_t0 = None
				pass
		else:
			self.ttv = None
			self.ttv_err = None
			self.epochs = None
			self.mean_ephem = None
			self.avg_period = None
			self.avg_t0 = None
		if n_per is not None:
			self.n_per = n_per
		else:
			self.n_per = 100

		self.prior_dict = {'mass_prior': mass_prior,
				'period_prior': period_prior,
				'h_prime_prior': h_prime_prior,
				'k_prime_prior': k_prime_prior,
				'inc_prior': inc_prior,
				'longnode_prior': longnode_prior}
		if self.transiting:
			if t0_prior is None:
				self.avg_t0 = self.ttv[0] - \
					self.epochs[0]*self.avg_period
				self.prior_dict['t0_prior'] = ('Uniform',
					self.avg_t0 - \
						self.n_per*self.ttv_err[0],
					self.avg_t0 + \
						self.n_per*self.ttv_err[0])
			else:
				self.prior_dict['t0_prior'] = t0_prior
		else:
			self.prior_dict['meananom_prior'] = meananom_prior

	def get_trend(self, data, obsind, errs):
		"""
		Compute a mean ephemeris for the planet.

		Parameters
		__________
		data : `~numpy.ndarray` with shape `(len_ttv,)`
			The timing data for the system.
		obsind : `~numpy.ndarray` with shape `(len_ttv,)`
			The epochs at which those timing data were taken.
		errs : `~numpy.ndarray` with shape `(len_ttv,)`
			The error on each timing data point.
		"""   
		z = np.polyfit(obsind, data, 1, w = 1/errs)
		p = np.poly1d(z)
		return p

	def validate_input(self, ttv, ttv_err, epochs):
		"""
		Validate the input for a transiting planet. Ensures that
		the timing, error, and epoch arrays are all set for a
		transiting planet or are all none for a non-transiting planet.
		Additionally ensures that the transiting planet arrays are 
		all the same shape.

		Parameters
		__________
		ttv : `~numpy.ndarray` with shape `(len_ttv,)`
			The timing data for the system.
		ttv_err : `~numpy.ndarray` with shape `(len_ttv,)`
			The error on each timing data point.
		epochs : `~numpy.ndarray` with shape `(len_ttv,)`
			The epochs at which those timing data were taken.
		"""   
	

		if ttv is None and ttv_err is None and epochs is None:
			return False
		elif ttv is not None and ttv_err is not None and \
			epochs is not None:
			l1 = len(ttv)
			l2 = len(ttv_err)
			l3 = len(epochs)
			if l1 == l2 == l3:
				return True
			else:
				raise ValueError('Shape mismatch between ' +
					f'lengths of TTVs {l1}, ' +
					f'TTV errors {l2}, and epochs {l3}')

		raise ValueError("Must specify TTVs, errors, and epochs!")
