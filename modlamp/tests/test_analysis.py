import unittest
from ..analysis import Global
from ..core import read_fasta
from os.path import abspath, join


class TestAnalysis(unittest.TestCase):
    sequences, _ = read_fasta(join(abspath('.'), 'modlamp/tests/files/lib.fasta'))
    a = Global(sequences)
    
    def test_libshape(self):
        self.assertEqual(self.a.library.shape[0], 246)
        
    def test_aa_freq(self):
        self.a.calc_aa_freq()
        self.assertAlmostEqual(self.a.aafreq[0, 0], 0.08250071)
        

if __name__ == '__main__':
    unittest.main()
