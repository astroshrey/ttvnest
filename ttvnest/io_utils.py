import dill as pickle

def save_results(system, outname = 'results.p'):
	with open(outname, 'wb') as f:
		pickle.dump(system, f)
	return outname

def load_results(loadname):
	system = pickle.load(open(loadname, 'rb'))
	return system
