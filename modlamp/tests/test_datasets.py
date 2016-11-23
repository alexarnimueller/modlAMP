import unittest
from ..datasets import load_helicalAMPset, load_AMPvsTMset, load_ACPvsNeg, load_AMPvsUniProt


class TestHelicalSet(unittest.TestCase):

    data = load_helicalAMPset()

    def test_sequences(self):
        self.assertEqual('NPATLMMFFK', self.data.sequences[3])

    def test_targets(self):
        self.assertEqual([0, 0, 1, 1], [int(self.data.target[i]) for i in [0, 362, 363, 725]])


class TestTMSet(unittest.TestCase):

    data = load_AMPvsTMset()

    def test_sequences(self):
        self.assertEqual('HGSIGAGVDW', self.data.sequences[5])

    def test_targets(self):
        self.assertEqual([0, 0, 1, 1], [int(self.data.target[i]) for i in [0, 205, 206, 411]])


class TestACPNeg(unittest.TestCase):
    
    data = load_ACPvsNeg()

    def test_sequences(self):
        self.assertEqual('GLFGVLAKVAAHVVPAIAEHF', self.data.sequences[123])

    def test_targets(self):
        self.assertEqual([0, 0, 1, 1], [int(self.data.target[i]) for i in [0, 93, 94, 188]])
        

class TestAMPUniport(unittest.TestCase):
    
    data = load_AMPvsUniProt()
    
    def test_sequences(self):
        self.assertEqual('GILSLVKGVAKLAGKGLAKEGGKFGLELIACKIAKQC', self.data.sequences[45])

    def test_targets(self):
        self.assertEqual([1, 1, 0, 0], [int(self.data.target[i]) for i in [0, 1608, 1609, 5887]])

if __name__ == '__main__':
    unittest.main()
