import numpy as np
import matplotlib.pyplot as plt
import ttvfast
from constants import *

def run_simulation(stellarmass, dt, sim_length, *argv):
	nplanets = len(argv)
	if nplanets < 2:
		print("You need at least 2 planets for a TTV simulation!")
		return None
	start = tkep - tref
	end = start + sim_length
	system = []
	for arg in argv:
		mp_ms, P, ecosw, esinw, M = arg
		e = np.sqrt(ecosw**2 + esinw**2)
		w = np.arctan2(esinw, ecosw)*180./np.pi
		planet_params = [mp_ms*earth_per_sun, P, e, 90., 0., w, M]
		system.append(ttvfast.models.Planet(*planet_params))	
	results = ttvfast.ttvfast(system, stellarmass, start, dt, end)
	results = results['positions']
	planet = np.array(results[0])
	time = np.array(results[2])
	timing_arrays = []
	for i in range(nplanets):
		timing_arr = []
		for plan, t in zip(planet, time):
			if plan == i and t != -2.0:
				timing_arr.append(t - start)
		timing_arrays.append(np.array(timing_arr, dtype = 'float'))
	return np.array(timing_arrays)
