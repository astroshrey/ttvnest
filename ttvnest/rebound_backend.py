import numpy as np
import rebound
from .constants import *

def setup_rebound_system(system, theta):
	start = 0.
	length = system.sim_length
	sim = rebound.Simulation()
	sim.dt = system.dt
	sim.add(m = 1.) #adding the star

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
		sim.add(m = mp_ms*earth_per_sun, P = P, e = e, inc = i, 
			Omega = Omega, omega = w, M = M)
	return sim

def track_resonant_angle(system, theta, p, q, sim_length,
	resolution_factor = 5):
	tmax = 2.*np.pi*sim_length #in rebound units
	resolution = 2.*np.pi*resolution_factor*system.dt/365.25
	t_samples = np.arange(0, tmax, resolution)
	sim = setup_rebound_system(system, theta)
	angles = []
	for t in t_samples:
		sim.integrate(t)
		l1 = sim.particles[1].l
		l2 = sim.particles[2].l
		varpi2 = sim.particles[2].pomega
		angles.append((p+q)*l2 - p*l1 - q*varpi2)
	return np.array(t_samples/2./np.pi), np.array(angles)
		



