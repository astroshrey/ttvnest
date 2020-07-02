import numpy as np
import rebound
from .constants import *
import multiprocessing
from tqdm import tqdm

def setup_rebound_system(system, theta):
	start = 0.
	sim = rebound.Simulation()
	sim.units = ('yr', 'AU', 'Msun')
	sim.add(m = 1.) #adding the star
	sim.integrator = "whfast"
	sim.dt = system.dt/365.25
	thetas = system.parse_with_fixed(theta)
	for i_pl, arr in enumerate(thetas):
		if system.planets[i_pl].transiting:
			mp_ms, P, hprime, kprime, i, Omega, t0 = arr
			e = hprime**2 + kprime**2
			w = np.arctan2(kprime, hprime)*180./np.pi
			t0 += start
			M = system.t0_to_mean_anom(t0, e, w, P, start)
		else:
			mp_ms, P, hprime, kprime, i, Omega, M = arr
	
		e = hprime**2 + kprime**2
		w = np.arctan2(kprime, hprime)*180./np.pi
		sim.add(m = mp_ms*earth_per_sun, P = P/365.25, e = e, inc = i, 
			Omega = Omega, omega = w, M = M)
	sim.move_to_com()
	return sim

def instability_calculator(pars):
	system, theta, sim_length = pars
	sim = setup_rebound_system(system, theta)
	sim.exit_max_distance = 20.
	times = np.linspace(0, sim_length, 1000)
	for i, time in enumerate(times):
		try:
			sim.integrate(time, exact_finish_time = 0)
		except (rebound.Escape, rebound.Collision, rebound.Encounter):
			return 0.
	return 1.

def megno_calculator(pars):
	system, theta, sim_length = pars
	sim = setup_rebound_system(system, theta)
	sim.init_megno()
	sim.exit_max_distance = 20.
	times = np.linspace(0, sim_length, 10000)
	for i, time in enumerate(times):
		try:
			sim.integrate(time, exact_finish_time = 0)
			megno = sim.calculate_megno()
			if megno > 10:
				return 10
		except (rebound.Escape, rebound.Collision, rebound.Encounter,
			rebound.SimulationError):
			return 10.
	return max(megno, 0) 

def calculate_MEGNOs(system, nsamps = 100, timescale = 1e3):
	inputs = []
	results = []
	array = np.random.permutation(system.samples_equal)[:nsamps]
	for samp in array:
		inputs.append((system, samp, timescale))

	for inp in tqdm(inputs):
		results.append(megno_calculator(inp))

	return np.array(results)

def calculate_instability(system, nsamps = 100, timescale = 1e6):
	inputs = []
	results = []
	array = np.random.permutation(system.samples_equal)[:nsamps]
	for samp in array:
		inputs.append((system, samp, timescale))
	for inp in tqdm(inputs):
		results.append(instability_calculator(inp))
	results = np.array(results, dtype = 'float')
	stable = np.sum(results)
	perc_stable = stable/nsamps*100.
	t_myr = timescale / 1e6
	print(f'{perc_stable}% of {nsamps} samples were stable on {t_myr} ' +
		'Myr timescale')
	return array, results

def track_resonant_angle(system, theta, p, q, sim_length,
	resolution_factor = 5):
	tmax = sim_length
	resolution = resolution_factor*(system.dt/365.25)
	t_samples = np.arange(0, tmax, resolution)
	sim = setup_rebound_system(system, theta)
	angles = []
	for t in t_samples:
		sim.integrate(t)
		l1 = sim.particles[1].l
		l2 = sim.particles[2].l
		varpi2 = sim.particles[2].pomega
		angles.append((p+q)*l2 - p*l1 - q*varpi2)
	return np.array(t_samples), np.array(angles)
		



