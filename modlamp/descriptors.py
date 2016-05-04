# -*- coding: utf-8 -*-
"""
.. module:: modlamp.descriptors

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates different classes to calculate peptide descriptor values. The following classes are available:

=============================		============================================================================
Class								Characteristics
=============================		============================================================================
:py:class:`GlobalDescriptor`		Global one-dimensional peptide descriptors calculated from the AA sequence.
:py:class:`PeptideDescriptor`		AA scale based global or convoluted descriptors (auto-/cross-correlated).
=============================		============================================================================

"""

import os
import sys
from core import load_scale, read_fasta, save_fasta, filter_unnatural, filter_values, filter_aa_more
import collections
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from Bio.SeqUtils.ProtParam import ProteinAnalysis

__author__ = 'modlab'
__docformat__ = "restructuredtext en"

class GlobalDescriptor(object):
	"""
	Base class for global, non-amino acid scale dependant descriptors. The following descriptors can be calculated by
	the **methods** linked below:

	- `Sequence Charge 		<modlamp.html#modlamp.descriptors.GlobalDescriptor.calculate_charge>`_
	- `Molecular Weight		<modlamp.html#modlamp.descriptors.GlobalDescriptor.calculate_MW>`_
	- `Sequence Length		<modlamp.html#modlamp.descriptors.GlobalDescriptor.length>`_
	- `Isoelectric Point	<modlamp.html#modlamp.descriptors.GlobalDescriptor.isoelectric_point>`_
	- `Charge Density		<modlamp.html#modlamp.descriptors.GlobalDescriptor.charge_density>`_
	- `Hydrophobic Ratio	<modlamp.html#modlamp.descriptors.GlobalDescriptor.hydrophobic_ratio>`_
	- `Aromaticity			<modlamp.html#modlamp.descriptors.GlobalDescriptor.aromaticity>`_
	- `Boman Index			<modlamp.html#modlamp.descriptors.GlobalDescriptor.boman_index>`_
	- `Aliphatic Index		<modlamp.html#modlamp.descriptors.GlobalDescriptor.aliphatic_index>`_
	- `Instability Index	<modlamp.html#modlamp.descriptors.GlobalDescriptor.instability_index>`_

	Most of the methods calculate values with help of the :mod:`Bio.SeqUtils.ProtParam` module of `Biopython <http://biopython.org/>`_.
	"""

	def __init__(self, seqs):
		"""
		:param seqs: a .fasta file with sequences, a list of sequences or a single sequence as string to calculate the descriptor values for.
		:return: initialized lists self.sequences, self.names and dictionary self.AA with amino acid scale values
		:Example:

		>>> P = GlobalDescriptor('KLAKLAKKLAKLAK')
		>>> P.sequences
		['KLAKLAKKLAKLAK']
		"""
		D = PeptideDescriptor(seqs, 'eisenberg')
		self.sequences = D.sequences
		self.names = D.names
		self.descriptor = D.descriptor
		self.target = D.target

	def length(self, append=False):
		"""
		Method to calculate the length (total AA count) of every sequence in the attribute :py:attr:`sequences`.

		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: array of descriptor values in the attribute :py:attr:`descriptor`
		"""
		desc = []
		for seq in self.sequences:
			desc.append(ProteinAnalysis(seq).length)
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def calculate_MW(self, amide=False, append=False):
		"""Method to calculate the molecular weight [g/mol] of every sequence in the attribute :py:attr:`sequences`.

		:param amide: {boolean} whether the sequences are C-terminally amidated (subtracts 0.95 from the MW).
		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: array of descriptor values in the attribute :py:attr:`descriptor`

		.. versionchanged:: v2.1.5 amide option added
		"""
		desc = []
		for seq in self.sequences:
			desc.append(ProteinAnalysis(seq).molecular_weight())
		desc = np.asarray(desc).reshape(len(desc), 1)
		if amide:  # if sequences are amidated, subtract 0.98 from calculated MW
			desc = [d-0.95 for d in desc]
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def _charge(self, seq, pH=7.0, amide=False):
		"""
		Calculates charge of a single sequence. Adapted from Bio.SeqUtils.IsoelectricPoint.IsoelectricPoint_chargeR function.
		The method used is first described by Bjellqvist. In the case of amidation, the value for the 'Cterm' pKa is 15 (and
		Cterm is added to the pos_pKs dictionary.
		The pKa scale is extracted from: http://www.hbcpnetbase.com/ (CRC Handbook of Chemistry and Physics, 96th edition).
		For further references, see the `Biopython <http://biopython.org/>`_ module :mod:`Bio.SeqUtils.IsoelectricPoint`.`

			pos_pKs = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}
			neg_pKs = {'Cterm': 2.15, 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}

		:param pH: {float} pH at which to calculate peptide charge.
		:param amide: {boolean} whether the sequences have an amidated C-terminus.
		:return: {array} descriptor values in the attribute :py:attr:`descriptor
		"""

		if amide:
			pos_pKs = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}
			neg_pKs = {'Cterm': 15., 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}
		else:
			pos_pKs = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}
			neg_pKs = {'Cterm': 2.15, 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}

		aa_content = ProteinAnalysis(seq).count_amino_acids()
		aa_content['Nterm'] = 1.0
		aa_content['Cterm'] = 1.0
		PositiveCharge = 0.0
		for aa, pK in pos_pKs.items():
			CR = 10 ** (pK - pH)
			partial_charge = CR / (CR + 1.0)
			PositiveCharge += aa_content[aa] * partial_charge
		NegativeCharge = 0.0
		for aa, pK in neg_pKs.items():
			CR = 10 ** (pH - pK)
			partial_charge = CR / (CR + 1.0)
			NegativeCharge += aa_content[aa] * partial_charge
		return PositiveCharge - NegativeCharge

	def calculate_charge(self, pH=7.0, amide=False, append=False):
		"""
		Method to overall charge of every sequence in the attribute :py:attr:`sequences`.
		Adapted from Bio.SeqUtils.IsoelectricPoint.IsoelectricPoint_chargeR function.

		The method used is first described by Bjellqvist. In the case of amidation, the value for the 'Cterm' pKa is 15 (and
		Cterm is added to the pos_pKs dictionary.
		The pKa scale is extracted from: http://www.hbcpnetbase.com/ (CRC Handbook of Chemistry and Physics, 96th edition).
		For further references, see the `Biopython <http://biopython.org/>`_ module :mod:`Bio.SeqUtils.IsoelectricPoint`.

			pos_pKs = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}
			neg_pKs = {'Cterm': 2.15, 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}

		:param pH: {float} pH at which to calculate peptide charge.
		:param amide: {boolean} whether the sequences have an amidated C-terminus.
		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: {array} descriptor values in the attribute :py:attr:`descriptor`
		"""

		desc = []
		for seq in self.sequences:
			desc.append(self._charge(seq, pH, amide))
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def charge_density(self, pH=7.0, amide=False, append=False):
		"""Method to calculate the charge density (charge / MW) of every sequences in the attributes :py:attr:`sequences`

		:param pH: {float} pH at which to calculate peptide charge.
		:param amide: {boolean} whether the sequences have an amidated C-terminus.
		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: array of descriptor values in the attribute :py:attr:`descriptor`.
		"""
		self.calculate_charge(pH, amide)
		charges = self.descriptor
		self.calculate_MW(amide)
		masses = self.descriptor
		desc = charges / masses
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def isoelectric_point(self, amide=False, append=False):
		"""
		Method to calculate the isoelectric point of every sequence in the attribute :py:attr:`sequences`.
		The pK scale is extracted from: http://www.hbcpnetbase.com/ (CRC Handbook of Chemistry and Physics, 96th edition).
		The method used is based on the IsoelectricPoint module in `Biopython <http://biopython.org/>`_
		module :mod:`Bio.SeqUtils.ProtParam`.

			pos_pKs = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}
			neg_pKs = {'Cterm': 2.15, 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}

		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: array of descriptor values in the attribute :py:attr:`descriptor`
		"""

		desc = []
		for seq in self.sequences:

			# Bracket between pH1 and pH2
			pH = 7.0
			charge = self._charge(seq, pH, amide)
			if charge > 0.0:
				pH1 = pH
				charge1 = charge
				while charge1 > 0.0:
					pH = pH1 + 1.0
					charge = self._charge(seq, pH, amide)
					if charge > 0.0:
						pH1 = pH
						charge1 = charge
					else:
						pH2 = pH
						charge2 = charge
						break
			else:
				pH2 = pH
				charge2 = charge
				while charge2 < 0.0:
					pH = pH2 - 1.0
					charge = self._charge(seq, pH, amide)
					if charge < 0.0:
						pH2 = pH
						charge2 = charge
					else:
						pH1 = pH
						charge1 = charge
						break
			# Bisection
			while pH2 - pH1 > 0.0001 and charge != 0.0:
				pH = (pH1 + pH2) / 2.0
				charge = self._charge(seq, pH, amide)
				if charge > 0.0:
					pH1 = pH
					charge1 = charge
				else:
					pH2 = pH
					charge2 = charge
			desc.append(pH)
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def instability_index(self, append=False):
		"""
		Method to calculate the instability of every sequence in the attribute :py:attr:`sequences`.
		The instability index is a prediction of protein stability based on the amino acid composition.
		([1] K. Guruprasad, B. V Reddy, M. W. Pandit, Protein Eng. 1990, 4, 155–161.)

		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: array of descriptor values in the attribute :py:attr:`descriptor`
		"""
		desc = []
		for seq in self.sequences:
			desc.append(ProteinAnalysis(seq).instability_index())
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def aromaticity(self, append=False):
		"""
		Method to calculate the aromaticity of every sequence in the attribute :py:attr:`sequences`.
		According to Lobry, 1994, it is simply the relative frequency of Phe+Trp+Tyr.

		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: array of descriptor values in the attribute :py:attr:`descriptor`
		"""
		desc = []
		for seq in self.sequences:
			desc.append(ProteinAnalysis(seq).aromaticity())
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def aliphatic_index(self, append=False):
		"""
		Method to calculate the aliphatic index of every sequence in the attribute :py:attr:`sequences`.
		According to Ikai, 1980, the aliphatic index is a measure of thermal stability of proteins and is dependant
		on the relative volume occupied by aliphatic amino acids (A,I,L & V).
		([1] A. Ikai, J. Biochem. 1980, 88, 1895–1898.)

		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: array of descriptor values in the attribute :py:attr:`descriptor`
		"""
		desc = []
		for seq in self.sequences:
			d = ProteinAnalysis(seq).count_amino_acids()
			d = {k:(float(d[k]) / len(seq)) * 100 for k in d.keys()}  # get mole percent of all AA
			desc.append(d['A'] + 2.9 * d['V'] + 3.9 * (d['I'] + d['L']))  # formula for calculating the AI (Ikai, 1980)
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def boman_index(self, append=False):
		"""
		Method to calculate the boman index of every sequence in the attribute :py:attr:`sequences`.
		According to Boman, 2003, the boman index is a measure for protein-protein interactions and is calculated by
		summing over all amino acid free energy of transfer [kcal/mol] between water and cyclohexane,[2] followed by
		dividing by	sequence length.
		([1] H. G. Boman, D. Wade, I. a Boman, B. Wåhlin, R. B. Merrifield, FEBS Lett. 1989, 259, 103–106.
		[2] A. Radzicka, R. Wolfenden, Biochemistry 1988, 27, 1664–1670.)

		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: array of descriptor values in the attribute :py:attr:`descriptor`
		"""
		D = {'L':-4.92,'I':-4.92,'V':-4.04,'F':-2.98,'M':-2.35,'W':-2.33,'A':-1.81,'C':-1.28,'G':-0.94,'Y':0.14,'T':2.57,
			 'S':3.40,'H':4.66,'Q':5.54,'K':5.55,'N':6.64,'E':6.81,'D':8.72,'R':14.92}
		desc = []
		for seq in self.sequences:
			val = []
			for a in seq:
				val.append(D[a])
			desc.append(sum(val)/len(val))
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def hydrophobic_ratio(self, append=False):
		"""
		Method to calculate the hydrophobic ratio of every sequence in the attribute :py:attr:`sequences`, which is the relative
		frequency of the amino acids [A,C,F,I,L,M & V].

		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
		:return: array of descriptor values in the attribute :py:attr:`descriptor`
		"""
		desc = []
		for seq in self.sequences:
			pa = ProteinAnalysis(seq).count_amino_acids()
			# formula for calculating the AI (Ikai, 1980):
			desc.append((pa['A'] + pa['C'] + pa['F'] + pa['I'] + pa['L'] + pa['M'] + pa['V']) / float(len(seq)))
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def feature_scaling(self, type='standard', fit=True):
		"""Method for feature scaling of the calculated descriptor matrix.

		:param type: **'standard'** or **'minmax'**, type of scaling to be used
		:param fit: {boolean}, defines whether the used scaler is first fitting on the data (True) or
			whether the already fitted scaler in :py:attr:`scaler` should be used to transform (False).
		:return: scaled descriptor values in :py:attr`self.descriptor`
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
		"""Method for shuffling features randomly.

		:return: descriptor matrix with shuffled feature columns in the attribute :py:attr:`descriptor`
		:Example:

		>>> D.descriptor
		array([[0.80685625,167.05234375,39.56818125,-0.26338667,155.16888667,33.48778]])
		>>> D.feature_shuffle()
		array([[155.16888667,-0.26338667,167.05234375,0.80685625,39.56818125,33.48778]])
		"""
		self.descriptor = shuffle(self.descriptor.transpose()).transpose()

	def filter_values(self, values, operator='=='):
		"""Method to filter the descriptor matrix in the attribute :py:attr:`descriptor` for a given list of values (same
		size as the number of features in the descriptor matrix!) The operator option tells the method whether to
		filter for values equal, lower, higher ect. to the given values in the **values** array.

		:param values: List/array of values to filter
		:param operator: filter criterion, available are all SQL like operators: ``==``, ``<``, ``>``, ``<=``and ``>=``.
		:return: filtered descriptor matrix and updated sequences in the corresponding attributes.

		.. seealso:: :func:`modlamp.core.filter_values()`
		"""
		filter_values(self, values, operator)

	def filter_aa(self, aminoacids=['X']):
		"""Method to filter sequences and corresponding descriptor values, if the sequences contain any of the given
		amino acids in the argument list **aminoacids**.

		:param aminoacids: List/array of amino acids to filter for
		:return: filtered descriptor matrix and updated sequences and names in the corresponding attributes.

		.. seealso:: :func:`modlamp.core.filter_aa_more()`
		"""
		filter_aa_more(self, aminoacids)

	def load_descriptordata(self, filename, delimiter=",", targets=False):
		"""Method to load any data file with sequences and descriptor values and save it to a new insatnce of the
		class :class:`modlamp.descriptors.PeptideDescriptor`.

		.. note::
			The data file should **not** have any headers

		:param filename: filename of the data file to be loaded
		:param delimiter: column delimiter
		:param targets: {boolean} whether last column in the file contains a target class vector
		:return: loaded sequences, descriptor values and targets in the corresponding attributes.
		"""
		data = np.genfromtxt(filename, delimiter=delimiter)
		data = data[:, 1:]  # skip sequences as they are "nan" when read as float
		seqs = np.genfromtxt(filename, delimiter=delimiter, dtype="str")
		seqs = seqs[:, 0]
		if targets:
			self.target = np.array(data[:, -1], dtype='int')
		self.sequences = seqs
		self.descriptor = data

	def save_descriptor(self, filename, delimiter=',', targets=[]):
		"""Method to save the descriptor values to a .csv/.txt file

		:param filename: filename of the output file
		:param delimiter: column delimiter
		:param targets: target class vector to be added to descriptor (same length as :py:attr:`sequences`)
		:return: output file with peptide names and descriptor values
		"""
		names = np.array(self.sequences, dtype='|S80')[:, np.newaxis]
		if len(targets) == len(self.sequences):
			target = np.array(targets)[:, np.newaxis]
			data = np.hstack((names, self.descriptor, target))
		else:
			data = np.hstack((names, self.descriptor))
		np.savetxt(filename, data, delimiter=delimiter, fmt='%s')


class PeptideDescriptor(object):
	"""
	Base class for peptide descriptors. The following **amino acid descriptor scales** are available for descriptor calculation:

	- **AASI**			(An amino acid selectivity index scale for helical antimicrobial peptides, *[1] D. Juretić, D. Vukicević, N. Ilić, N. Antcheva, A. Tossi, J. Chem. Inf. Model. 2009, 49, 2873–2882.*)
	- **argos**			(Argos hydrophobicity amino acid scale, *[2] P. Argos, J. K. M. Rao, P. A. Hargrave, Eur. J. Biochem. 2005, 128, 565–575.*)
	- **bulkiness**		(Amino acid side chain bulkiness scale, *[3] J. M. Zimmerman, N. Eliezer, R. Simha, J. Theor. Biol. 1968, 21, 170–201.*)
	- **charge_physio**	(Amino acid charge at pH 7.0 - Hystidine charge +0.1.)
	- **charge_acidic**	(Amino acid charge at acidic pH - Hystidine charge +1.0.)
	- **cougar**		(modlabs inhouse selection of global peptide descriptors)
	- **eisenberg**		(the Eisenberg hydrophobicity consensus amino acid scale, *[4] D. Eisenberg, R. M. Weiss, T. C. Terwilliger, W. Wilcox, Faraday Symp. Chem. Soc. 1982, 17, 109.*)
	- **Ez** 			(potential that assesses energies of insertion of amino acid side chains into lipid bilayers, *[5] A. Senes, D. C. Chadi, P. B. Law, R. F. S. Walters, V. Nanda, W. F. DeGrado, J. Mol. Biol. 2007, 366, 436–448.*)
	- **flexibility**	(amino acid side chain flexibilitiy scale, *[6] R. Bhaskaran, P. K. Ponnuswamy, Int. J. Pept. Protein Res. 1988, 32, 241–255.*)
	- **gravy**			(GRAVY hydrophobicity amino acid scale, *[7] J. Kyte, R. F. Doolittle, J. Mol. Biol. 1982, 157, 105–132.*)
	- **hopp-woods**	(Hopp-Woods amino acid hydrophobicity scale,*[8] T. P. Hopp, K. R. Woods, Proc. Natl. Acad. Sci. 1981, 78, 3824–3828.*)
	- **ISAECI**		(Isotropic Surface Area (ISA) and Electronic Charge Index (ECI) of amino acid side chains, *[9] E. R. Collantes, W. J. Dunn, J. Med. Chem. 1995, 38, 2705–2713.*)
	- **janin** 		(Janin hydrophobicity amino acid scale, [10] J. L. Cornette, K. B. Cease, H. Margalit, J. L. Spouge, J. A. Berzofsky, C. DeLisi, J. Mol. Biol. 1987, 195, 659–685.*)
	- **kytedoolittle**	(Kyte & Doolittle hydrophobicity amino acid scale, *[11] J. Kyte, R. F. Doolittle, J. Mol. Biol. 1982, 157, 105–132.*)
	- **Levitt_alpha**	(Levitt amino acid alpha-helix propensity scale, extracted from http://web.expasy.org/protscale. *[12] M. Levitt, Biochemistry 1978, 17, 4277-4285.*)
	- **MSS**			(A graph-theoretical index that reflects topological shape and size of amino acid side chains, *[13] C. Raychaudhury, A. Banerjee, P. Bag, S. Roy, J. Chem. Inf. Comput. Sci. 1999, 39, 248–254.*)
	- **MSW**			(Amino acid scale based on a PCA of the molecular surface based WHIM descriptor (MS-WHIM), extended to natural amino acids, *[14] A. Zaliani, E. Gancia, J. Chem. Inf. Comput. Sci 1999, 39, 525–533.*)
	- **pepcats**		(modlabs pharmacophoric feature based PEPCATS scale, *[15] C. P. Koch, A. M. Perna, M. Pillong, N. K. Todoroff, P. Wrede, G. Folkers, J. A. Hiss, G. Schneider, PLoS Comput. Biol. 2013, 9, e1003088.*)
	- **polarity**		(Amino acid polarity scale, *[3] J. M. Zimmerman, N. Eliezer, R. Simha, J. Theor. Biol. 1968, 21, 170–201.*)
	- **PPCALI**		(modlabs inhouse scale derived from a PCA of 143 amino acid property scales, *[15] C. P. Koch, A. M. Perna, M. Pillong, N. K. Todoroff, P. Wrede, G. Folkers, J. A. Hiss, G. Schneider, PLoS Comput. Biol. 2013, 9, e1003088.*)
	- **refractivity**	(Relative amino acid refractivity values, *[16] T. L. McMeekin, M. Wilensky, M. L. Groves, Biochem. Biophys. Res. Commun. 1962, 7, 151–156.*)
	- **t_scale**		(A PCA derived scale based on amino acid side chain properties calculated with 6 different probes of the GRID program, *[17] M. Cocchi, E. Johansson, Quant. Struct. Act. Relationships 1993, 12, 1–8.*)
	- **TM_tend**		(Amino acid transmembrane propensity scale, extracted from http://web.expasy.org/protscale, *[18] Zhao, G., London E. Protein Sci. 2006, 15, 1987-2001.*)
	- **z3**			(The original three dimensional Z-scale, *[17] S. Hellberg, M. Sjöström, B. Skagerberg, S. Wold, J. Med. Chem. 1987, 30, 1126–1135.*)
	- **z5**			(The extended five dimensional Z-scale, *[18] M. Sandberg, L. Eriksson, J. Jonsson, M. Sjöström, S. Wold, J. Med. Chem. 1998, 41, 2481–2491.*)

	Further, amino acid scale independent methods can be calculated with help of the :class:`GlobalDescriptor` class.

	"""

	def __init__(self, seqs, scalename='eisenberg'):
		"""
		:param seqs: a .fasta file with sequences, a list of sequences or a single sequence as string to calculate the descriptor values for.
		:param scalename: name of the amino acid scale (one of the given list above) used to calculate the descriptor values
		:return: initialized attributes :py:attr:`sequences`, :py:attr:`names` and dictionary :py:attr:`scale` with amino acid scale values of the scale name in :py:attr:`scalename`.
		:Example:

		>>> AMP = PeptideDescriptor('KLLKLLKKLLKLLK','pepcats')
		>>> AMP.sequences
		['KLLKLLKKLLKLLK']
		"""
		if type(seqs) == list:
			self.sequences = seqs
			self.names = []
		elif type(seqs) == np.ndarray:
			self.sequences = seqs.tolist()
			self.names = []
		elif type(seqs) == str and seqs.isupper():
			self.sequences = [seqs]
			self.names = []
		elif os.path.isfile(seqs):
			self.sequences, self.names = read_fasta(seqs)
		else:
			print "'inputfile' does not exist, is not a valid list of sequences or is not a valid sequence string"

		self.scalename = scalename
		self.scale = load_scale(self.scalename)
		self.descriptor = np.array([[]])
		self.target = np.array([], dtype='int')

	def load_scale(self, scalename):
		"""Method to load amino acid values from a given scale

		:param scalename: name of the amino acid scale to be loaded.
		:return: loaded amino acid scale values in a dictionary in the attribute :py:attr:`scale`.

		.. seealso:: :func:`modlamp.core.load_scale()`
		"""
		self.scale = load_scale(scalename)

	def read_fasta(self, filename):
		"""Method for loading sequences from a FASTA formatted file into the attributes :py:attr:`sequences` and
		:py:attr:`names`. This method is used by the base class :class:`PeptideDescriptor` if the input is a FASTA file.

		:param filename: .fasta file with sequences and headers to read
		:return: list of sequences in self.sequences with corresponding sequence names in self.names
		"""
		self.sequences, self.names = read_fasta(filename)

	def save_fasta(self, outputfile):
		"""Method for saving sequences from :py:attr:`sequences` to a FASTA formatted file.

		:param outputfile: filename of the output FASTA file
		:return: list of sequences in self.sequences with corresponding sequence names in the attribute :py:attr:`names`
		"""
		save_fasta(self,outputfile)

	def calculate_autocorr(self, window, append=False):
		"""Method for auto-correlating the amino acid values for a given descriptor scale

		:param window: correlation window for descriptor calculation in a sliding window approach
		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
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
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def calculate_crosscorr(self, window, append=False):
		"""Method for cross-correlating the amino acid values for a given descriptor scale

		:param window: correlation window for descriptor calculation in a sliding window approach
		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
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
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def calculate_moment(self, window=1000, angle=100, modality='max', append=False):
		"""Method for calculating the maximum or mean moment of the amino acid values for a given descriptor scale and window.

		:param window: {int} amino acid window in which to calculate the moment. If the sequence is shorter than the window, the length of the sequence is taken. So if the default window of 1000 is chosen, for all sequences shorter than 1000, the **global** hydrophobic moment will be calculated. Otherwise, the maximal hydrophiobic moment for the chosen window size found in the sequence will be returned.
		:param angle: {int} angle in which to calculate the moment. **100** for alpha helices, **180** for beta sheets.
		:param modality: {'max' or 'mean'} Calculate respectively maximum or mean hydrophobic moment.
		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
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
		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def calculate_global(self, window=1000, modality='max', append=False):
		"""Method for calculating a global / window averaging descriptor value of a given AA scale

		:param window: {int} amino acid window in which to calculate the moment. If the sequence is shorter than the window, the length of the sequence is taken.
		:param modality: {'max' or 'mean'} Calculate respectively maximum or mean hydrophobic moment.
		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
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

		desc = np.asarray(desc).reshape(len(desc), 1)
		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def calculate_profile(self, type='uH', window=7, append=False):
		"""Method for calculating hydrophobicity or hydrophobic moment profiles for given sequences and fitting for slope and intercept. The hydrophobicity scale used is "eisenberg"

		:param type: type of profile, available: 'H' for hydrophobicity or 'uH' for hydrophobic moment
		:param window: {int} size of sliding window used (odd-numbered).
		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the attribute :py:attr:`descriptor`.
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

		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def count_aa(self, scale='relative', append=False):
		"""Method for producing the amino acid distribution for the given sequences as a descriptor

		:param scale: {'absolute' or 'relative'} defines whether counts or frequencies are given for each AA
		:param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
			attribute :py:attr:`descriptor`.
		:return: the amino acid distributions for every sequence individually in the attribute :py:attr:`descriptor`
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

		if append:
			self.descriptor = np.hstack((self.descriptor, np.array(desc)))
		else:
			self.descriptor = np.array(desc)

	def feature_scaling(self, type='standard', fit=True):
		"""Method for feature scaling of the calculated descriptor matrix.

		:param type: {'standard' or 'minmax'} type of scaling to be used
		:param fit: {boolean} defines whether the used scaler is first fitting on the data (True) or
			whether the already fitted scaler in :py:attr:`scaler` should be used to transform (False).
		:return: scaled descriptor values in :py:attr:`descriptor`
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
		"""Method for shuffling feature columns randomly.

		:return: descriptor matrix with shuffled feature columns in :py:attr:`descriptor`
		:Example:

		>>> D.descriptor
		array([[0.80685625,167.05234375,39.56818125,-0.26338667,155.16888667,33.48778]])
		>>> D.feature_shuffle()
		array([[155.16888667,-0.26338667,167.05234375,0.80685625,39.56818125,33.48778]])
		"""
		self.descriptor = shuffle(self.descriptor.transpose()).transpose()

# TODO move to core for both sequences and descriptors

	def sequence_order_shuffle(self):
		"""Method for shuffling sequence order in self.sequences.

		:return: sequences in :py:attr`self.sequences` with shuffled order in the list.
		:Example:

		>>> D.sequences
		['LILRALKGAARALKVA','VKIAKIALKIIKGLG','VGVRLIKGIGRVARGAI','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV']
		>>> D.sequence_order_shuffle()
		>>> D.sequences
		['VGVRLIKGIGRVARGAI','LILRALKGAARALKVA','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV','VKIAKIALKIIKGLG']
		"""
		self.sequences = shuffle(self.sequences)

	def filter_unnatrual(self):
		"""Method to filter out sequences with unnatural amino acids from :py:attr:`sequences` as well as duplicates.
		:return: Filtered sequence list in the attribute :py:attr:`sequences`
		"""
		filter_unnatural(self)

	def filter_values(self, values, operator='=='):
		"""Method to filter the descriptor matrix in the attribute :py:attr:`descriptor` for a given list of value (same
		size as the number of features in the descriptor matrix!)

		:param values: List of values to filter
		:param operator: filter criterion, available are all SQL like operators: ``==``, ``<``, ``>``, ``<=``and ``>=``.
		:return: filtered descriptor matrix and updated sequences in the corresponding attributes.

		.. seealso:: :func:`modlamp.core.filter_values()`
		"""
		filter_values(self, values, operator)

	def filter_aa(self, aminoacids=['X']):
		"""Method to filter sequences and corresponding descriptor values, if the sequences contain any of the given
		amino acids in the argument list **aminoacids**.

		:param aminoacids: List/array of amino acids to filter for
		:return: filtered descriptor matrix and updated sequences and names in the corresponding attributes.

		.. seealso:: :func:`modlamp.core.filter_aa_more()`
		"""
		filter_aa_more(self, aminoacids)

	def load_descriptordata(self, filename, delimiter=",", targets=False):
		"""Method to load any data file with sequences and descriptor values and save it to a new insatnce of the
		class :class:`modlamp.descriptors.PeptideDescriptor`.

		.. note::
			The data file should **not** have any headers

		:param filename: filenam of the data file to be loaded
		:param delimiter: column delimiter
		:param targets: {boolean} whether last column in the file contains a target class vector
		:return: loaded sequences, descriptor values and targets in the corresponding attributes.
		"""
		data = np.genfromtxt(filename, delimiter=delimiter)
		data = data[:, 1:]  # skip sequences as they are "nan" when read as float
		seqs = np.genfromtxt(filename, delimiter=delimiter, dtype="str")
		seqs = seqs[:, 0]
		if targets:
			self.target = np.array(data[:, -1], dtype='int')
		self.sequences = seqs
		self.descriptor = data

	def save_descriptor(self, filename, delimiter=',', targets=None):
		"""Method to save the descriptor values to a .csv/.txt file

		:param filename: filename of the output file
		:param delimiter: column delimiter
		:param targets: target class vector to be added to descriptor (same length as :py:attr:`sequences`)
		:return: output file with peptide names and descriptor values
		"""
		names = np.array(self.sequences, dtype='|S80')[:, np.newaxis]
		if len(targets) == len(self.sequences):
			target = np.array(targets)[:, np.newaxis]
			data = np.hstack((names, self.descriptor, target))
		else:
			data = np.hstack((names, self.descriptor))
		np.savetxt(filename, data, delimiter=delimiter, fmt='%s')
