import numpy as np
from constants import *

def get_data(catfile, koi_star, nplanets):
	obs = []
	errs = []
	for i in range(nplanets):
		ob, err = pull_ttvs(catfile, koi_star + .01*(i+1))
		obs.append(ob)
		errs.append(err)
	return np.array(obs), np.array(errs)

def pull_ttvs(catfile, koi):
	f = open(catfile, 'r')
	times = []
	errs = []
	ttvs = []
	for i in range(n_comment_lines):
		f.readline()
	for line in f:
		data = line.split()
		cur_koi = float(data[0])
		if cur_koi > koi:
			break
		if cur_koi == koi:
			calculated = float(data[2])
			ttv = float(data[3])/60./24. #in days
			err = float(data[4])/60./24. #in days
			observed = ttv + calculated
			flag = data[-1]
			if flag != '*':
				times.append(observed)
				ttvs.append(ttv)
				errs.append(err)
	return np.array(times), np.array(errs)
