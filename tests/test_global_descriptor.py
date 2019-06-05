import unittest
import numpy as np
from os.path import dirname, join

from modlamp.descriptors import GlobalDescriptor


__author__ = 'modlab'


class TestGlobalDescriptor(unittest.TestCase):

    G = GlobalDescriptor(['GLFDIVKKVVGALG', 'LLLLLL', 'KKKKKKKKKK', 'DDDDDDDDDDDD'])
    G2 = GlobalDescriptor(join(dirname(__file__), 'files/lib.fasta'))
    G3 = GlobalDescriptor(join(dirname(__file__), 'files/lib.csv'))

    def test_load(self):
        self.assertEqual('GLFDIVKKVVGALG', self.G.sequences[0])
        self.assertEqual('LASKSTSGIGVFGRIRAGLKLKST', self.G2.sequences[2])
        self.assertEqual('NPGKSTTRRI', self.G3.sequences[-1])

    def test_charge(self):
        self.G.calculate_charge()
        self.assertAlmostEqual(self.G.descriptor[0, 0], 0.996, 3)
        self.G.calculate_charge(amide=True)
        self.assertAlmostEqual(self.G.descriptor[0, 0], 1.996, 3)
        self.G.calculate_charge(ph=9.84)
        self.assertAlmostEqual(self.G.descriptor[0, 0], -0.000, 3)

    def test_isoelectric(self):
        self.G.isoelectric_point()
        self.assertAlmostEqual(self.G.descriptor[0, 0], 9.840, 3)
        self.G.isoelectric_point(amide=True)
        self.assertAlmostEqual(self.G.descriptor[0, 0], 10.7090, 4)

    def test_charge_density(self):
        self.G.charge_density()
        self.assertAlmostEqual(self.G.descriptor[0, 0], 0.00070, 4)
        self.G.charge_density(amide=True)
    
    def test_aliphatic_index(self):
        self.G.aliphatic_index()
        self.assertAlmostEqual(self.G.descriptor[0, 0], 152.857, 3)

    def test_boman_index(self):
        self.G.boman_index()
        self.assertAlmostEqual(self.G.descriptor[0, 0], -1.0479, 4)

    def test_filter_aa(self):
        D = GlobalDescriptor(['GLFDIVKKVVGALG', 'LLLLLL', 'KKKKKKKKKK', 'DDDDDDDDDDDD'])
        D.calculate_charge()
        D.filter_aa(['D'])
        self.assertEqual(D.sequences, ['LLLLLL', 'KKKKKKKKKK'])
        self.assertEqual(len(D.descriptor), 2)

    def test_filter_values(self):
        E = GlobalDescriptor(['GLFDIVKKVVGALG', 'LLLLLL', 'KKKKKKKKKK', 'DDDDDDDDDDDD'])
        E.calculate_charge()
        E.filter_values(values=[1.], operator='>=')
        self.assertEqual(E.sequences, ['KKKKKKKKKK'])
        self.assertEqual(len(E.descriptor), 1)

    def test_instability_index(self):
        self.G.instability_index()
        self.assertAlmostEqual(self.G.descriptor[0, 0], -8.214, 3)

    def test_length(self):
        self.G.length()
        self.assertEqual(self.G.descriptor[0, 0], 14)

    def test_molweight(self):
        self.G.calculate_MW()
        self.assertEqual(self.G.descriptor[0, 0], 1415.72)

    def test_featurescaling(self):
        self.G.calculate_charge()
        self.G.calculate_MW(append=True)
        self.G.feature_scaling()
        self.assertAlmostEqual(-5.55111512e-17, np.mean(self.G.descriptor, axis=0).tolist()[0])
        self.assertAlmostEqual(1., np.std(self.G.descriptor, axis=0).tolist()[0])

    def test_hydroratio(self):
        self.G.hydrophobic_ratio()
        self.assertAlmostEqual(0.57142857, self.G.descriptor[0][0])
        
    def test_aromaticity(self):
        self.G.aromaticity()
        self.assertAlmostEqual(0.07142857142857142, self.G.descriptor[0][0])

    def test_formula(self):
        self.G.formula(amide=True, append=True)
        self.assertEqual('C67 H115 N17 O16', self.G.descriptor[0, -1])
