import unittest
from modlamp.datasets import load_helicalAMPset, load_AMPvsTMset, load_ACPvsTM, load_ACPvsRandom, load_AMPvsUniProt


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


class TestACPvsTM(unittest.TestCase):
    
    data = load_ACPvsTM()

    def test_sequences(self):
        self.assertEqual('NAVGTGVMGGMVTATVLAIFF', self.data.sequences[600])

    def test_targets(self):
        self.assertEqual([1, 1, 0, 0], [self.data.target[i] for i in [0, 412, 413, 825]])

    def test_n_sequences(self):
        self.assertEqual(826, len(self.data.sequences))

class TestACPvsRandom(unittest.TestCase):

    data = load_ACPvsRandom()

    def test_sequences(self):
        self.assertEqual('KVTYLLLEGGK', self.data.sequences[600])

    def test_targets(self):
        self.assertEqual([1, 1, 0, 0], [self.data.target[i] for i in [0, 412, 413, 825]])

    def test_n_sequences(self):
        self.assertEqual(826, len(self.data.sequences))
        

class TestAMPUniport(unittest.TestCase):
    
    data = load_AMPvsUniProt()
    
    def test_sequences(self):
        self.assertEqual('GILSLVKGVAKLAGKGLAKEGGKFGLELIACKIAKQC', self.data.sequences[45])

    def test_targets(self):
        self.assertEqual([1, 1, 0, 0], [int(self.data.target[i]) for i in [0, 1608, 1609, 5887]])

if __name__ == '__main__':
    unittest.main()
