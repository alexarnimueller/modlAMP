import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('.'))

from modlamp.descriptors import GlobalDescriptor

__author__ = 'modlab'


class TestGlobalDescriptor(unittest.TestCase):

	G = GlobalDescriptor(['GLFDIVKKVVGALG', 'LLLLLL', 'KKKKKKKKKK', 'DDDDDDDDDDDD'])

	def test_charge(self):
		self.G.calculate_charge()
		self.assertAlmostEqual(self.G.descriptor[0], 0.9959478)
		self.G.calculate_charge(amide=True)
		self.assertAlmostEqual(self.G.descriptor[0], 1.9959337)
		self.G.calculate_charge(pH=9.84)
		self.assertAlmostEqual(self.G.descriptor[0], -0.0002392)

	def test_isoelectric(self):
		self.G.isoelectric_point()
		self.assertAlmostEqual(self.G.descriptor[0], 9.8397827)
		self.G.isoelectric_point(amide=True)
		self.assertAlmostEqual(self.G.descriptor[0], 10.7089233)

	def test_molweight(self):
		self.G.calculate_MW()
		self.assertAlmostEqual(self.G.descriptor[0], 1415.71869999)

	def test_length(self):
		self.G.length()
		self.assertEqual(self.G.descriptor[0], 14)

	def test_charge_density(self):
		self.G.charge_density()
		self.assertAlmostEqual(self.G.descriptor[0], 0.00070349)
		self.G.charge_density(amide=True)
		self.assertAlmostEqual(self.G.descriptor[0], 0.00141078)

	def test_instability_index(self):
		self.G.instability_index()
		self.assertAlmostEqual(self.G.descriptor[0], -8.21428571)

	def test_aliphatic_index(self):
		self.G.aliphatic_index()
		self.assertAlmostEqual(self.G.descriptor[0], 152.85714286)

	def test_boman_index(self):
		self.G.boman_index()
		self.assertAlmostEqual(self.G.descriptor[0], -1.04785714)

	def test_filter_values(self):
		self.G.calculate_charge()
		self.G.filter_values(values=[1.], operator='>=')
		self.assertEqual(self.G.sequences, ['GLFDIVKKVVGALG', 'KKKKKKKKKK'])
		self.assertEqual(len(self.G.descriptor), 2)

	def test_filter_aa(self):
		D = GlobalDescriptor(['GLFDIVKKVVGALG','LLLLLL','KKKKKKKKKK','DDDDDDDDDDDD'])
		D.calculate_charge()
		D.filter_aa(['D'])
		self.assertEqual(D.sequences, ['LLLLLL', 'KKKKKKKKKK'])
		self.assertEqual(len(D.descriptor), 2)
