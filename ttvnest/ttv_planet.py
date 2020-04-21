import numpy as np
from . import plot_utils as pu
from .constants import *
import dynesty
from dynesty import utils as dyfunc

class TTVPlanet:
	def __init__(self, ttv = None, ttv_err = None, epochs = None,
		mass_prior = ('Uniform', 0., 100.),
		period_prior = ('Uniform', 1., 100.),
		h_prime_prior = ('Uniform', -1., 1.), 
		k_prime_prior = ('Uniform', -1., 1.),
		inc_prior = ('Fixed', 90.),
		longnode_prior = ('Fixed', 0.),
		t0_prior = None,
		meananom_prior = ('Periodic', 0., 360.)):
		"""
		Docstring
		"""
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
					self.avg_t0 - 100*self.ttv_err[0],
					self.avg_t0 + 100*self.ttv_err[0])
			else:
				self.prior_dict['t0_prior'] = t0_prior
		else:
			self.prior_dict['meananom_prior'] = meananom_prior

	def get_trend(self, data, obsind, errs):
		z = np.polyfit(obsind, data, 1, w = 1/errs)
		p = np.poly1d(z)
		return p

	def validate_input(self, ttv, ttv_err, epochs):
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
				raise ValueError(f'Shape mismatch between lengths of TTVs {l1}, TTV errors {l2}, and epochs {l3}')

		raise ValueError("Must specify TTVs, errors, and epochs!")
