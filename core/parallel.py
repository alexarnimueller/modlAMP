"""
.. module:: parallel

.. moduleauthor:: modlab Alex Mueller <alex.mueller@pharma.ethz.ch>

Helper functions to parallelize functions.
"""

import numpy as np
from functools import partial

def parallel_function(f):
	"""
	Function to parallelize a given function.

	:param f: name of a function to be parallelized
	:return: callable parallelized function
	:Example:

	>>> somefunction_parallel = parallel_function(somefunction)
	>>> parallel_result = somefunction_parallel(input_for_somefunction)
	"""

	def _easy_parallize(f, sequence):
		"""
		Assumes the given function f takes a sequence as an input.

		:param f: function to parallelize
		:param sequence: input for function f
		:return: specific function that can now be called and parallelizes
		"""
		from multiprocessing import Pool
		pool = Pool(processes=4) # depends on available cores
		result = pool.map(f, sequence) # for i in sequence: result[i] = f(i)
		cleaned = [x for x in result if not x is None] # getting results
		cleaned = np.asarray(cleaned)
		pool.close() # not optimal! but easy
		pool.join()
		return cleaned

	return partial(_easy_parallize, f)