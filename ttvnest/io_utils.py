import pickle

def save_results(results, outname = 'results.p'):
	pickle.dump(results, open(outname, 'wb'))
	return outname

def load_results(loadname):
	results = pickle.load(open(loadname, 'rb'))
	return results
