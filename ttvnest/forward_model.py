import numpy as np
import matplotlib.pyplot as plt
import ttvfast
from .constants import *

def run_simulation(stellarmass, dt, start_time, sim_length, theta,
	fixed_params, nplanets, transiting):

	start = start_time - tref
	end = start + sim_length

	system = []
	paramv = parse_with_fixed(theta, fixed_params, nplanets)
	for param in paramv:
		mp_ms, P, ecosw, esinw, i, Omega, t0 = param
		e = np.sqrt(ecosw**2 + esinw**2)
		w = np.arctan2(esinw, ecosw)*180./np.pi
		t0 += start
		M = t0_to_mean_anom(t0, e, w, P, start)
		planet_params = [mp_ms*earth_per_sun, P, e, i, Omega, w, M]
		system.append(ttvfast.models.Planet(*planet_params))	
	results = ttvfast.ttvfast(system, stellarmass, start, dt, end)
	results = results['positions']
	planet = np.array(results[0], dtype = 'int')
	time = np.array(results[2], dtype = 'float')
	timing_arrays = []
	for i in transiting:
		timing_arr = time[planet == i]
		timing_arr = timing_arr[timing_arr != -2.0]
		timing_arr -= start
		timing_arrays.append(timing_arr)
	return np.array(timing_arrays)

def parse_fixed(params, fixed_params, planet_tag):
	for fixed in fixed_params:
		if fixed[0] == planet_tag:
			params[param_id(fixed[1])] = fixed[2]
	return params

def t0_to_mean_anom(t0, e, w_deg, P, t_epoch):
	w = w_deg * np.pi / 180. #omega in radians
	f_0 = np.pi/2 - w 	#true anomaly at inf conj	
	E_0 = 2*np.arctan(np.tan(f_0/2) * np.sqrt((1-e)/(1+e)))
	M_0 = E_0 - e*np.sin(E_0)
	#above are eccentric and mean anomalies at inf conj

	t_peri = t0 - P*M_0/(2*np.pi)
	mean_anomaly = 360./P*(t_epoch - t_peri)
	return mean_anomaly % 360

def parse_with_fixed(theta, fixed_params, nplanets):
	paramv = []
	current_ind = 0
	for i in range(nplanets):
		theta_init = np.zeros(7) - 1
		for fixed in fixed_params:
			if fixed[0] == i:
				theta_init[param_id(fixed[1])] = fixed[2]
		remaining = np.argwhere(theta_init == -1).flatten()
		for ind in remaining:
			theta_init[ind] = theta[current_ind]
			current_ind += 1
		paramv.append(theta_init)
	return paramv

def param_id(varname):
	if varname == 'mass':
		return 0
	elif varname == 'period':
		return 1
	elif varname == 'ecosw':
		return 2
	elif varname == 'esinw':
		return 3
	elif varname == 'inc':
		return 4
	elif varname == 'LongNode':
		return 5
	elif varname == 'MeanAnom':
		return 6
	return None
