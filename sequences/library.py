"""
.. module:: library

.. moduleauthor:: modlab Alex Mueller <alex.mueller@pharma.ethz.ch>

This module hosts the class :class:`MixedLibrary()` to generate a virtual peptide library composed out of different
sublibraries. The available library subtypes are all from the modules :mod:`centrosymmetric`,
:mod:`helix`, :mod:`kinked`, :mod:`oblique` or :mod:`random_seq`.
"""

from centrosymmetric import CentroSequences
from helix import Helices
from kinked import Kinked
from oblique import Oblique
from random_seq import RandomSeqs
from sklearn.utils import shuffle
from core.templates import filter_unnatural

class MixedLibrary:
	"""
	Base class for holding a virtual peptide library.
	"""

	def __init__(self,number,centrosymmetric=1,centroasymmetric=1,helix=1,kinked=1,oblique=1,rand=1,randAMP=1,randAMPnoCM=1):
		"""
		init method of the class :class:`MixedLibrary`. Except from **number**, all other parameters are ratios of
		sequences of the given sequence class.

		:param number: number of sequences to be generated
		:param centrosymmetric: ratio of symmetric centrosymmetric sequences in the library
		:param centroasymmetric: ratio of asymmetric centrosymmetric sequences in the library
		:param helix: ratio of amphipathic helical sequences in the library
		:param kinked: ratio of kinked amphipathic helical sequences in the library
		:param oblique: ratio of oblique oriented amphipathic helical sequences in the library
		:param random: ratio of random sequneces in the library
		:param randAMP: ratio of random sequences with APD2 amino acid distribution in the library
		:param randAMPnoCM: ratio of random sequences with APD2 amino acid distribution without Cys and Met in the library
		"""
		self.sequences = []
		self.number = int(number)
		self.norm = float(sum((centrosymmetric,centroasymmetric,helix,kinked,oblique,rand,randAMP,randAMPnoCM)))
		self.ratio_centrosym = float(centrosymmetric) / self.norm
		self.ratio_centroasym = float(centroasymmetric) / self.norm
		self.ratio_helix = float(helix) / self.norm
		self.ratio_kinked = float(kinked) / self.norm
		self.ratio_oblique = float(oblique) / self.norm
		self.ratio_rand = float(rand) / self.norm
		self.ratio_randAMP = float(randAMP) / self.norm
		self.ratio_randAMPnoCM = float(randAMPnoCM) / self.norm


	def generate_library(self):
		"""
		This method generates a virtual sequence library with the subtype ratios initialized in class :class:`MixedLibrary()`.
		All sequences are between 7 and 28 amino acids in length.

		:return: a virtual library of sequences in self.sequences
		:Example:

		>>> Lib = MixedLibrary(10000,centrosymmetric=5,centroasymmetric=5,helix=3,kinked=3,oblique=2,rand=10,randAMP=10,randAMPnoCM=5)
		>>> Lib.generate_library()
		>>> len(Lib.sequences)
		10000
		>>> Lib.sequences
		['RHTHVAGSWYGKMPPSPQTL','MRIKLRKIPCILAC','DGINKEVKDSYGVFLK','LRLYLRLGRVWVRG','GKLFLKGGKLFLKGGKLFLKG',...]
		>>> Lib.ratio_helix
		0.069767
		"""
		Cs = CentroSequences(round(float(self.number) * self.ratio_centrosym, ndigits=0))
		Cs.generate_symmetric()
		Ca = CentroSequences(round(float(self.number) * self.ratio_centroasym, ndigits=0))
		Ca.generate_asymmetric()
		H = Helices(7,28,round(float(self.number) * self.ratio_helix, ndigits=0))
		H.generate_helices()
		K = Kinked(7,28,round(float(self.number) * self.ratio_kinked, ndigits=0))
		K.generate_kinked()
		O = Oblique(7,28,round(float(self.number) * self.ratio_oblique, ndigits=0))
		O.generate_oblique()
		R = RandomSeqs(7,28,round(float(self.number) * self.ratio_rand, ndigits=0))
		R.generate_sequences('rand')
		Ra = RandomSeqs(7,28,round(float(self.number) * self.ratio_randAMP, ndigits=0))
		Ra.generate_sequences('AMP')
		Rc = RandomSeqs(7,28,round(float(self.number) * self.ratio_randAMPnoCM, ndigits=0))
		Rc.generate_sequences('AMPnoCM')

		self.sequences = Cs.sequences + Ca.sequences + H.sequences + K.sequences + O.sequences + R.sequences + Ra.sequences + Rc.sequences
		self.sequences = shuffle(self.sequences)

		# check if rounding affected sequence number. if too many: chop end off, if too few: fill up with random seqs
		if len(self.sequences) > self.number:
			self.sequences = self.sequences[:-(len(self.sequences)-self.number)]
		elif len(self.sequences) < self.number:
			S = RandomSeqs(7,28,self.number - len(self.sequences))
			S.generate_sequences()
			self.sequences = self.sequences + S.sequences


	def filter_unnatrual(self):
		"""
		Method to filter out sequences with unnatural amino acids from :py:attr:`self.sequences` as well as duplicates.
		:return: Filtered sequence list in :py:attr:`self.sequences`
		"""
		filter_unnatural(self)