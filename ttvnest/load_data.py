import numpy as np
from astroquery.vizier import Vizier

def get_data(koi_star, nplanets, datatable = 'J/ApJS/235/38/table6'):
	obs = []
	errs = []
	print("Downloading data from Vizier...")
	Vizier.ROW_LIMIT = -1 #full catalog
	cats = Vizier.get_catalogs(datatable)
	data = cats[0]
	if not (koi_star + .01 in data['KOI']):
		print("Can't find this KOI in the Thompson+18 catalog")
		print("Falling back on Rowe+15 catalog...")
		cats = Vizier.get_catalogs('J/ApJS/217/16/table2')
		data = cats[0]

	for i in range(nplanets):
		cur_koi = koi_star + .01*(i + 1)
		cur_dat = data[data['KOI']==cur_koi]
		calculated = np.array(cur_dat['tn'], dtype = float)
		ttv = np.array(cur_dat['TTVn'], dtype = float)
		err = np.array(cur_dat['e_TTVn'], dtype = float)
		obs.append(calculated + ttv)
		errs.append(err)
	print("Data retrieved!")
	return np.array(obs), np.array(errs)
