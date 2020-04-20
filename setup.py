import setuptools

setuptools.setup(
	name = "ttvnest",
	url = "https://github.com/astroshrey/ttvnest",
	version = '0.0.1',
	author = "Shreyas Vissapragada",
	author_email = "svissapr@caltech.edu",
	packages = ["ttvnest"],
	license = "MIT",
	intall_requires=["numpy", "matplotlib", "ttvfast-python",
		"dynesty", "dill", "scipy", "astropy", "astroquery"]
	)
