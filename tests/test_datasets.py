import unittest
from modlamp.datasets import load_AMPvsTM, load_ACPvsTM, load_ACPvsRandom, load_AMPvsUniProt, load_custom


class TestTMSet(unittest.TestCase):

    data = load_AMPvsTM()

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
        self.assertEqual('GNNRPVYIPQPRPPHPRI', self.data.sequences[5])

    def test_targets(self):
        self.assertEqual([1, 1, 0, 0], [int(self.data.target[i]) for i in [0, 2599, 2600, 5199]])


class TestCustom(unittest.TestCase):

    data = load_custom('custom.csv')

    def test_sequences(self):
        self.assertEqual('GTLEFDVTIGRAN', self.data.sequences[0])

    def test_targets(self):
        self.assertEqual([1, 0], [int(self.data.target[i]) for i in [0, 3]])


class TestBunch(unittest.TestCase):

    data = load_AMPvsTM()

    def test_setattr(self):
        self.data.test = ['test']

    def test_keyerror(self):
        self.assertRaises(AttributeError, self.data.__getattr__, 'notthere')

    def test_setstate(self):
        self.data.__setstate__(1)


if __name__ == '__main__':
    unittest.main()
