import numpy as np
from astroquery.vizier import Vizier
from astropy.io import ascii

def get_data(koi_star, nplanets, datatable = 'J/ApJS/217/16/table2'):
	obs = []
	errs = []
	epochs = []
	print("Downloading Rowe+15 data from Vizier...")
	Vizier.ROW_LIMIT = 10000 #full catalog
	cats = Vizier.query_constraints(catalog = datatable,
		KOI = '>'+str(koi_star)+' & <' + str(koi_star+1))
	data = cats[0]

	for i in range(nplanets):
		cur_koi = koi_star + .01*(i + 1)
		cur_dat = data[data['KOI']==cur_koi]
		epoch = np.array(cur_dat['n'], dtype = int) - 1 #zero ind 
		calculated = np.array(cur_dat['tn'], dtype = float)
		ttv = np.array(cur_dat['TTVn'], dtype = float)
		err = np.array(cur_dat['e_TTVn'], dtype = float)
		obs.append(calculated + ttv)
		errs.append(err)
		epochs.append(epoch)
	print("Data retrieved!")
	return np.array(obs), np.array(errs), np.array(epochs)

