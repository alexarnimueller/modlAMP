# -*- coding: utf-8 -*-
"""
.. module:: peptide_descriptor

.. moduleauthor:: modlab Alex Mueller <alex.mueller@pharma.ethz.ch>
"""

import os
import sys
from core.templates import load_scale, read_fasta, save_fasta
import collections
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from Bio.SeqUtils.ProtParam import ProteinAnalysis

__author__ = 'modlab'


class PeptideDescriptor:
	"""
	Base class for peptide descriptors. The following **amino acid descriptor scales** are available for descriptor calculation:

	- **AASI**			(An amino acid selectivity index scale for helical antimicrobial peptides, *[1] D. Juretić, D. Vukicević, N. Ilić, N. Antcheva, A. Tossi, J. Chem. Inf. Model. 2009, 49, 2873–2882.*)
	- **argos**			(Argos hydrophobicity amino acid scale, *[2] P. Argos, J. K. M. Rao, P. A. Hargrave, Eur. J. Biochem. 2005, 128, 565–575.*)
	- **bulkiness**		(Amino acid side chain bulkiness scale, *[3] J. M. Zimmerman, N. Eliezer, R. Simha, J. Theor. Biol. 1968, 21, 170–201.*)
	- **cougar**		(modlabs inhouse selection of global peptide descriptors)
	- **eisenberg**		(the Eisenberg hydrophobicity consensus amino acid scale, *[4] D. Eisenberg, R. M. Weiss, T. C. Terwilliger, W. Wilcox, Faraday Symp. Chem. Soc. 1982, 17, 109.*)
	- **Ez** 			(potential that assesses energies of insertion of amino acid side chains into lipid bilayers, *[5] A. Senes, D. C. Chadi, P. B. Law, R. F. S. Walters, V. Nanda, W. F. DeGrado, J. Mol. Biol. 2007, 366, 436–448.*)
	- **flexibility**	(amino acid side chain flexibilitiy scale, *[6] R. Bhaskaran, P. K. Ponnuswamy, Int. J. Pept. Protein Res. 1988, 32, 241–255.*)
	- **gravy**			(GRAVY hydrophobicity amino acid scale, *[7] J. Kyte, R. F. Doolittle, J. Mol. Biol. 1982, 157, 105–132.*)
	- **hopp-woods**	(Hopp-Woods amino acid hydrophobicity scale,*[8] T. P. Hopp, K. R. Woods, Proc. Natl. Acad. Sci. 1981, 78, 3824–3828.*)
	- **ISAECI**		(Isotropic Surface Area (ISA) and Electronic Charge Index (ECI) of amino acid side chains, *[9] E. R. Collantes, W. J. Dunn, J. Med. Chem. 1995, 38, 2705–2713.*)
	- **janin** 		(Janin hydrophobicity amino acid scale, [10] J. L. Cornette, K. B. Cease, H. Margalit, J. L. Spouge, J. A. Berzofsky, C. DeLisi, J. Mol. Biol. 1987, 195, 659–685.*)
	- **kytedoolittle**	(Kyte & Doolittle hydrophobicity amino acid scale, *[11] J. Kyte, R. F. Doolittle, J. Mol. Biol. 1982, 157, 105–132.*)
	- **MSS**			(A graph-theoretical index that reflects topological shape and size of amino acid side chains, *[12] C. Raychaudhury, A. Banerjee, P. Bag, S. Roy, J. Chem. Inf. Comput. Sci. 1999, 39, 248–254.*)
	- **MSW**			(Amino acid scale based on a PCA of the molecular surface based WHIM descriptor (MS-WHIM), extended to natural amino acids, *[13] A. Zaliani, E. Gancia, J. Chem. Inf. Comput. Sci 1999, 39, 525–533.*)
	- **pepcats**		(modlabs pharmacophoric feature based PEPCATS scale, *[14] C. P. Koch, A. M. Perna, M. Pillong, N. K. Todoroff, P. Wrede, G. Folkers, J. A. Hiss, G. Schneider, PLoS Comput. Biol. 2013, 9, e1003088.*)
	- **polarity**		(Amino acid polarity scale, *[3] J. M. Zimmerman, N. Eliezer, R. Simha, J. Theor. Biol. 1968, 21, 170–201.*)
	- **PPCALI**		(modlabs inhouse scale derived from a PCA of 143 amino acid property scales, *[14] C. P. Koch, A. M. Perna, M. Pillong, N. K. Todoroff, P. Wrede, G. Folkers, J. A. Hiss, G. Schneider, PLoS Comput. Biol. 2013, 9, e1003088.*)
	- **refractivity**	(Relative amino acid refractivity values, *[15] T. L. McMeekin, M. Wilensky, M. L. Groves, Biochem. Biophys. Res. Commun. 1962, 7, 151–156.*)
	- **t_scale**		(A PCA derived scale based on amino acid side chain properties calculated with 6 different probes of the GRID program, *[16] M. Cocchi, E. Johansson, Quant. Struct. Act. Relationships 1993, 12, 1–8.*)
	- **z3**			(The original three dimensional Z-scale, *[17] S. Hellberg, M. Sjöström, B. Skagerberg, S. Wold, J. Med. Chem. 1987, 30, 1126–1135.*)
	- **z5**			(The extended five dimensional Z-scale, *[18] M. Sandberg, L. Eriksson, J. Jonsson, M. Sjöström, S. Wold, J. Med. Chem. 1998, 41, 2481–2491.*)

	Further, amino acid scale independent methods can be calculated with help of the :class:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor()` class.

	"""

	def __init__(self, inputfile, scalename='eisenberg'):
		"""
		:param inputfile: a .fasta file with sequences, a list of sequences or a single sequence as string to calculate the descriptor values for.
		:param scalename: name of the amino acid scale (one of the given list above) used to calculate the descriptor values
		:return: initialized lists self.sequences, self.names and dictionary self.AA with amino acid scale values
		:Example:

		>>> AMP = PeptideDescriptor('KLLKLLKKLLKLLK','pepcats')
		>>> AMP.sequences
		['KLLKLLKKLLKLLK']
		"""
		if type(inputfile) == list:
			self.sequences = inputfile
			self.names = []
		elif type(inputfile) == str and inputfile.isupper():
			self.sequences = [inputfile]
			self.names = []
		elif os.path.isfile(inputfile):
			self.read_fasta(inputfile)
		else:
			print "'inputfile' does not exist, is not a valid list of sequences or is not a valid sequence string"
			sys.exit()

		load_scale(self,scalename)


	def read_fasta(self, inputfile):
		"""
		Method for loading sequences from a FASTA formatted file into self.sequences & self.names. This method is
		used by the base class :class:`PeptideDescriptor` if the input is a FASTA file.

		:param inputfile: .fasta file with sequences and headers to read
		:return: list of sequences in self.sequences with corresponding sequence names in self.names
		"""
		read_fasta(self, inputfile)


	def save_fasta(self, outputfile):
		"""
		Method for saving sequences from :py:attr:`self.sequences` to a FASTA formatted file.

		:param outputfile: filename of the output FASTA file
		:return: list of sequences in self.sequences with corresponding sequence names in :py:attr:`self.names`
		"""
		save_fasta(self,outputfile)


	def calculate_autocorr(self, window):
		"""
		Method for auto-correlating the amino acid values for a given descriptor scale

		:param window: correlation window for descriptor calculation in a sliding window approach
		:return: calculated descriptor numpy.array in self.descriptor
		:Example:

		>>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL','PPCALI')
		>>> AMP.calculate_autocorr(7)
		>>> AMP.descriptor
		array([[  1.28442339e+00,   1.29025116e+00,   1.03240901e+00, .... ]])
		>>> AMP.descriptor.shape
		(1, 133)
		"""
		desc = list()
		for s in range(len(self.sequences)):  # iterate over all sequences
			seq = self.sequences[s]
			M = list()  # list of lists to store translated sequence values
			for l in range(len(seq)):  # translate AA sequence into values
				M.append(self.scale[str(seq[l])])

			# auto-correlation in defined sequence window
			seqdesc = list()
			for dist in range(window):  # for all correlation distances
				for val in range(len(self.scale['A'])):  # for all features of the descriptor scale
					valsum = list()
					cntr = 0.
					for pos in range(len(seq)):  # for every position in the sequence
						if (pos + dist) < len(seq):  # check if correlation distance is possible at that sequence position
							cntr += 1  # counter to scale sum
							valsum.append(M[pos][val] * M[pos + dist][val])
					seqdesc.append(sum(valsum) / cntr)  # append scaled correlation distance values

			desc.append(seqdesc)  # store final descriptor values in "descriptor"
		self.descriptor = np.array(desc)


	def calculate_crosscorr(self, window):
		"""
		Method for cross-correlating the amino acid values for a given descriptor scale

		:param window: correlation window for descriptor calculation in a sliding window approach
		:return: calculated descriptor numpy.array in self.descriptor
		:Example:

		>>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL','pepcats')
		>>> AMP.calculate_crosscorr(7)
		>>> AMP.descriptor
		array([[ 0.6875    ,  0.46666667,  0.42857143,  0.61538462,  0.58333333, ... ]])
		>>> AMP.descriptor.shape
		(1, 147)
		"""
		desc = list()
		for s in range(len(self.sequences)):  # iterate over all sequences
			seq = self.sequences[s]
			M = list()  # list of lists to store translated sequence values
			for l in range(len(seq)):  # translate AA sequence into values
				M.append(self.scale[str(seq[l])])

			# auto-correlation in defined sequence window
			seqdesc = list()
			for val in range(len(self.scale['A'])):  # for all features of the descriptor scale
				for cc in range(len(self.scale['A'])):  # for every feature cross correlation
					if (val + cc) < len(
							self.scale['A']):  # check if crosscorr distance is in range of the amount of features
						for dist in range(window):  # for all correlation distances
							cntr = float()
							valsum = list()
							for pos in range(len(seq)):  # for every position in the sequence
								if (pos + dist) < len(
										seq):  # check if correlation distance is possible at that sequence position
									cntr += 1  # counter to scale sum
									valsum.append(M[pos][val] * M[pos + dist][val + cc])
							seqdesc.append(sum(valsum) / cntr)  # append scaled correlation distance values

			desc.append(seqdesc)  # store final descriptor values in "descriptor"
		self.descriptor = np.asarray(desc)

	def calculate_moment(self, window=1000, angle=100, modality='max'):
		"""
		Method for calculating the maximum or mean moment of the amino acid values for a given descriptor scale and window.

		:param window: {int} amino acid window in which to calculate the moment. If the sequence is shorter than the window, the length of the sequence is taken. So if the default window of 1000 is chosen, for all sequences shorter than 1000, the **global** hydrophobic moment will be calculated. Otherwise, the maximal hydrophiobic moment for the chosen window size found in the sequence will be returned.
		:param angle: {int} angle in which to calculate the moment. **100** for alpha helices, **180** for beta sheets.
		:param modality: (max or mean) Calculate respectively maximum or mean hydrophobic moment.
		:return: Calculated descriptor as a numpy.array in self.descriptor and all possible global values in self.all_moms (needed for calculate_profile method)
		:Example:

		>>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL','eisenberg')
		>>> AMP.calculate_moment(window=1000, angle=100, modality='max')
		>>> AMP.descriptor
		array([[ 0.48790226]])
		"""
		if self.scale['A'] == list:
			print '\n Descriptor moment calculation is only possible for one dimensional descriptors.\n'
			sys.exit()

		desc = list()
		self.all_moms = list()
		for s, seq in enumerate(self.sequences):
			wdw = min(window, len(seq))  # if sequence is shorter than window, take the whole sequence instead
			M = list()
			for l in range(len(seq)):
				M.append(self.scale[str(seq[l])])

			Mwdw = list()
			for i in range(len(M) - wdw + 1):
				Mwdw.append(sum(M[i:i + wdw], []))

			Mwdw = np.asarray(Mwdw)
			rads = angle * (np.pi / 180) * np.asarray(range(wdw))  # calculate actual moment (radial)
			vcos = (Mwdw * np.cos(rads)).sum(axis=1)
			vsin = (Mwdw * np.sin(rads)).sum(axis=1)
			moms = np.sqrt(vsin * vsin + vcos * vcos) / wdw

			if modality == 'max':  # take window with maximal value
				moment = np.max(moms)
			elif modality == 'mean':  # take average value over all windows
				moment = np.mean(moms)
			else:
				print '\nModality parameter is wrong, please choose between "max" and "mean".\n'
				sys.exit()
			desc.append(moment)
			self.all_moms.append(moms)
		desc = np.asarray(desc)
		self.descriptor = desc.reshape(len(desc), 1)

	def calculate_global(self, window=1000, modality='max'):
		"""
		Method for calculating a global / window averaging descriptor value of a given AA scale

		:param window: {int} amino acid window in which to calculate the moment. If the sequence is shorter than the window, the length of the sequence is taken.
		:param modality: (max or mean) Calculate respectively maximum or mean hydrophobic moment.
		:return: Calculated descriptor as numpy.array in self.descriptor and all possible global values in self.all_globs (needed for calculate_profile method)
		:Example:

		>>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL','eisenberg')
		>>> AMP.calculate_global(window=1000, modality='max')
		>>> AMP.descriptor
		array([[ 0.44875]])
		"""
		desc = list()
		self.all_globs = list()
		for n, seq in enumerate(self.sequences):
			wdw = min(window, len(seq))
			M = list()
			for l in range(len(seq)):  # translate AA sequence into values
				M.append(self.scale[str(seq[l])])
			Mwdw = list()
			for i in range(len(M) - wdw + 1):
				Mwdw.append(sum(M[i:i + wdw],[]))  # list of all the values for the different windows
			Mwdw = np.asarray(Mwdw)
			glob = np.sum(Mwdw, axis=1) / wdw
			try:
				if modality == 'max':
					outglob = np.max(glob)  # returned moment will be the maximum of all windows
				elif modality == 'mean':
					outglob = np.mean(glob)  # returned moment will be the mean of all windows
				desc.append(outglob)
				self.all_globs.append(glob)

			except:
				print 'Modality parameter is wrong, please choose between "max" and "mean"\n.'



		desc = np.asarray(desc)
		self.descriptor = desc.reshape(len(desc), 1)

	def calculate_profile(self, type='uH', window=7):
		"""
		Method for calculating hydrophobicity or hydrophobic moment profiles for given sequences and fitting for slope and intercept. The hydrophobicity scale used is "eisenberg"

		:param type: type of profile, available: 'H' for hydrophobicity or 'uH' for hydrophobic moment
		:param window: {int} size of sliding window used (odd-numbered).
		:return: Fitted slope and intercept of calculated profile for every given sequence in self.descriptor
		:Example:

		>>> AMP = PeptideDescriptor('KLLKLLKKVVGALG','kytedoolittle')
		>>> AMP.calculate_profile(type='H')
		>>> AMP.descriptor
		array([[ 0.03731293,  0.19246599]])
		"""
		if type == 'uH':
			self.calculate_moment(window=window)
			self.y_vals = self.all_moms
		elif type == 'H':
			self.calculate_global(window=window)
			self.y_vals = self.all_globs
		else:
			print 'Type parameter is wrong, please choose between "uH" for hydrophobic moment and "H" for hydrophobicity\n.'
			sys.exit()

		desc = list()
		for n, seq in enumerate(self.sequences):
			self.x_vals = range(len(seq))[((window - 1) / 2):-((window - 1) / 2)]
			if len(seq) <= window:
				slope, intercept, r_value, p_value, std_err = [0, 0, 0, 0, 0]
			else:
				slope, intercept, r_value, p_value, std_err = stats.linregress(self.x_vals, self.y_vals[n])
			desc.append([slope, intercept])

		self.descriptor = np.asarray(desc)

	def count_aa(self, scale='relative'):
		"""
		Method for producing the amino acid distribution for the given sequences as a descriptor

		:scale: ('absolute' or 'relative') defines whether counts or frequencies are given for each AA
		:return: self.descriptor containing the amino acid distributions for every sequence individually
		:Example:

		>>> AMP = PeptideDescriptor('ACDEFGHIKLMNPQRSTVWY','pepcats') # aa_count() does not depend on the descriptor scale
		>>> AMP.count_aa()
		>>> AMP.descriptor
		array([[ 0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05, ... ]])
		>>> AMP.descriptor.shape
		(1, 20)
		"""
		desc = list()
		scl = 1
		for seq in self.sequences:
			if scale == 'relative':
				scl = len(seq)
			d = {a: (float(seq.count(a)) / scl) for a in self.scale.keys()}
			od = collections.OrderedDict(sorted(d.items()))
			desc.append(od.values())

		self.descriptor = np.asarray(desc)

# TODO: make modle "load_descriptor" to directly load saved values again

	def save_descriptor(self, filename, delimiter=','):
		"""
		Method to save the descriptor values to a .csv/.txt file

		:param filename: filename of the output file
		:param delimiter: column delimiter
		:return: output file with peptide names and descriptor values
		"""
		names = np.array(self.sequences, dtype='|S20')[:, np.newaxis]
		data = np.hstack((names, self.descriptor))
		np.savetxt(filename, data, delimiter=delimiter, fmt='%s')

# TODO: add possibility to save target vector into descriptor directly

	def feature_scaling(self,type='standard',fit=True):
		"""
		Method for feature scaling of the calculated descriptor matrix.

		:param type: **'standard'** or **'minmax'**, type of scaling to be used
		:param fit: **True** or **False**, defines whether the used scaler is first fitting on the data (True) or
		 whether the already fitted scaler in self.scaler should be used to transform (False).
		:return: scaled descriptor values in self.descriptor
		:Example:

		>>> D.descriptor
		array([[0.155],[0.34],[0.16235294],[-0.08842105],[0.116]])
		>>> D.feature_scaling(type='minmax',fit=True)
		array([[0.56818182],[1.],[0.5853447],[0.],[0.47714988]])
		"""
		try:
			if type == 'standard':
				self.scaler = StandardScaler()
			elif type =='minmax':
				self.scaler = MinMaxScaler()

			if fit == True:
				self.descriptor = self.scaler.fit_transform(self.descriptor)
			else:
				self.descriptor = self.scaler.transform(self.descriptor)
		except:
			print "Unknown scaler type!\nAvailable: 'standard', 'minmax'"


	def feature_shuffle(self):
		"""
		Method for shuffling features randomly.

		:return: descriptor matrix with shuffled feature columns in self.descriptor
		:Example:

		>>> D.descriptor
		array([[0.80685625,167.05234375,39.56818125,-0.26338667,155.16888667,33.48778]])
		>>> D.feature_shuffle()
		array([[155.16888667,-0.26338667,167.05234375,0.80685625,39.56818125,33.48778]])
		"""
		self.descriptor = shuffle(self.descriptor.transpose()).transpose()

# TODO move to core for both sequences and descriptors
	def sequence_order_shuffle(self):
		"""
		Method for shuffling sequence order in self.sequences.

		:return: sequences in self.sequences with shuffled order in the list.
		:Example:

		>>> D.sequences
		['LILRALKGAARALKVA','VKIAKIALKIIKGLG','VGVRLIKGIGRVARGAI','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV']
		>>> D.sequence_order_shuffle()
		>>> D.sequences
		['VGVRLIKGIGRVARGAI','LILRALKGAARALKVA','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV','VKIAKIALKIIKGLG']
		"""
		self.sequences = shuffle(self.sequences)

class GlobalDescriptor:
	"""
	Base class for global, non-amino acid scale dependant descriptors. The following descriptors can be calculated by
	the modules specified in brackets:

	- **Sequence Charge**	(:meth:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor.calculate_charge()`)
	- **Molecular Weight**	(:meth:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor.calculate_MW()`)
	- **Sequence Length**	(:meth:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor.length()`)
	- **Isoelectric Point**	(:meth:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor.isoelectric_point()`)
	- **Charge Density**	(:meth:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor.charge_density()`)
	- **Hydrophobic Ratio**	(not available yet)
	- **Aromaticity**		(:meth:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor.aromaticity()`)
	- **Boman Index**		(:meth:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor.boman_index()`)
	- **Aliphatic Index**	(:meth:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor.aliphatic_index()`)
	- **Instability Index**	(:meth:`modlAMP.descriptors.peptide_descriptor.GlobalDescriptor.instability_index()`)
	"""

	def __init__(self,inputfile):
		"""
		:param inputfile: a .fasta file with sequences, a list of sequences or a single sequence as string to calculate the descriptor values for.
		:return: initialized lists self.sequences, self.names and dictionary self.AA with amino acid scale values
		:Example:

		>>> P = GlobalDescriptor('KLAKLAKKLAKLAK')
		>>> P.sequences
		['KLAKLAKKLAKLAK']
		"""
		D = PeptideDescriptor(inputfile,'eisenberg')
		self.sequences = D.sequences
		self.names = D.names


	def isoelectric_point(self):
		"""
		Method to calculate the isoelectric point of every sequence in :py:attr:`self.sequences`.

		:return: array of descriptor values in :py:attr:`self.descriptor`
		"""
		desc = []
		for seq in self.sequences:
			desc.append(ProteinAnalysis(seq).isoelectric_point())
		self.descriptor = np.asarray(desc)


	def calculate_charge(self):
		"""
		Method to calculate the overall charge of every sequence in :py:attr:`self.sequences`.

		:return: array of descriptor values in :py:attr:`self.descriptor`

		The following dictionary shows the used side chain charges at neutral pH::

			AACharge = {"C":-.045,"D":-.999,"E":-.998,"H":.091,"K":1,"R":1,"Y":-.001}

		"""
		AACharge = {"C":-.045,"D":-.999,"E":-.998,"H":.091,"K":1,"R":1,"Y":-.001}
		desc = []
		for seq in self.sequences:
			charge = 0.
			for a in seq:
				charge += AACharge.get(a,0)
			desc.append(charge)
		self.descriptor = np.asarray(desc)


	def calculate_MW(self):
		"""
		Method to calculate the molecular weight [g/mol] of every sequence in :py:attr:`self.sequences`.

		:return: array of descriptor values in :py:attr:`self.descriptor`
		"""
		desc = []
		for seq in self.sequences:
			desc.append(ProteinAnalysis(seq).molecular_weight())
		self.descriptor = np.asarray(desc)


	def length(self):
		"""
		Method to calculate the length (total AA count) of every sequence in :py:attr:`self.sequences`.

		:return: array of descriptor values in :py:attr:`self.descriptor`
		"""
		desc = []
		for seq in self.sequences:
			desc.append(ProteinAnalysis(seq).length)
		self.descriptor = np.asarray(desc)


	def charge_density(self):
		"""
		Method to calculate the charge density (charge / MW) of every sequence in :py:attr:`self.sequences`.

		:return: array of descriptor values in :py:attr:`self.descriptor`
		"""
		desc = []
		self.calculate_charge()
		for i,seq in enumerate(self.sequences):
			desc.append(self.descriptor[i] / ProteinAnalysis(seq).molecular_weight())
		self.descriptor = np.asarray(desc)


	def instability_index(self):
		"""
		Method to calculate the instability of every sequence in :py:attr:`self.sequences`.
		The instability index is a prediction of protein stability based on the amino acid composition.
		([1] K. Guruprasad, B. V Reddy, M. W. Pandit, Protein Eng. 1990, 4, 155–161.)

		:return: array of descriptor values in :py:attr:`self.descriptor`
		"""
		desc = []
		for seq in self.sequences:
			desc.append(ProteinAnalysis(seq).instability_index())
		self.descriptor = np.asarray(desc)


	def aromaticity(self):
		"""
		Method to calculate the aromaticity of every sequence in :py:attr:`self.sequences`.
		According to Lobry, 1994, it is simply the relative frequency of Phe+Trp+Tyr.

		:return: array of descriptor values in :py:attr:`self.descriptor`
		"""
		desc = []
		for seq in self.sequences:
			desc.append(ProteinAnalysis(seq).aromaticity())
		self.descriptor = np.asarray(desc)


	def aliphatic_index(self):
		"""
		Method to calculate the aliphatic index of every sequence in :py:attr:`self.sequences`.
		According to Ikai, 1980, the aliphatic index is a measure of thermal stability of proteins and is dependant
		on the relative volume occupied by aliphatic amino acids (A,I,L & V).
		([1] A. Ikai, J. Biochem. 1980, 88, 1895–1898.)

		:return: array of descriptor values in :py:attr:`self.descriptor`
		"""
		desc = []
		for seq in self.sequences:
			D = ProteinAnalysis(seq).count_amino_acids()
			desc.append(D['A'] + 2.9 * D['V'] + 3.9 * (D['I'] + D['L'])) # formula for calculating the AI (Ikai, 1980)
		self.descriptor = np.asarray(desc)


	def boman_index(self):
		"""
		Method to calculate the boman index of every sequence in :py:attr:`self.sequences`.
		According to Boman, 2003, the boman index is a measure for protein-protein interactions and is calculated by
		summing over all amino acid free energy of transfer [kcal/mol] between water and cyclohexane,[2] followed by
		dividing by	sequence length.
		([1] H. G. Boman, D. Wade, I. a Boman, B. Wåhlin, R. B. Merrifield, FEBS Lett. 1989, 259, 103–106.
		[2] A. Radzicka, R. Wolfenden, Biochemistry 1988, 27, 1664–1670.)

		:return: array of descriptor values in :py:attr:`self.descriptor`
		"""
		D = {'L':-4.92,'I':-4.92,'V':-4.04,'F':-2.98,'M':-2.35,'W':-2.33,'A':-1.81,'C':-1.28,'G':-0.94,'Y':0.14,'T':2.57,
			 'S':3.40,'H':4.66,'Q':5.54,'K':5.55,'N':6.64,'E':6.81,'D':8.72,'R':14.92}
		desc = []
		for seq in self.sequences:
			val = []
			for a in seq:
				val.append(D[a])
			desc.append(sum(val)/len(val))
		self.descriptor = np.asarray(desc)


	def hydrophobic_ratio(self):
		"""
		Method to calculate the hydrophobic ratio of every sequence in :py:attr:`self.sequences`, which is the relative
		frequency of the amino acids [A,C,F,I,L,M & V].

		:return: array of descriptor values in :py:attr:`self.descriptor`
		"""
		desc = []
		for seq in self.sequences:
			D = ProteinAnalysis(seq).count_amino_acids()
			desc.append((D['A'] + D['C'] + D['F'] + D['I'] + D['L'] + D['M'] + D['V']) / float(len(seq))) # formula for calculating the AI (Ikai, 1980)
		self.descriptor = np.asarray(desc)

# TODO: test for most new methods!