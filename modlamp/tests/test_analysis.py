import unittest
from ..analysis import Global
from ..core import read_fasta
from os.path import abspath, join


class TestAnalysis(unittest.TestCase):
    sequences, _ = read_fasta(join(abspath('.'), 'modlamp/tests/files/lib.fasta'))
    a = Global(sequences)
    
    def test_libshape(self):
        self.assertEqual(self.a.library.shape[1], 246)
        
    def test_aa_freq(self):
        self.a.calc_aa_freq(plot=False)
        self.assertAlmostEqual(self.a.aafreq[0, 0], 0.08250071)
    
#    def test_H(self):
#        self.a.calc_H(plot=False)
#        self.assertAlmostEqual(self.a.H[0, :5], [0., 0., 0., 0., 0.,])
#        self.assertAlmostEqual(self.a.mean_H[0], 0.)
    
#    def test_uH(self):
#        pass
#
#    def test_charge(self):
#        pass
#
    def test_len(self):
        self.a.calc_len()
        self.assertEqual(self.a.len[0, 2], 24.)

if __name__ == '__main__':
    unittest.main()
