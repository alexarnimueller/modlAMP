import unittest
from modlamp.sequences import Random
from modlamp.descriptors import PeptideDescriptor


class TestCore(unittest.TestCase):
    sequences = ['GLFDIVKKVVGALG', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALK', 'ABCDEFGHAJKLMNOPQRSTUVWXYZ']
    r = Random(10, 20, 1)
    r.sequences = sequences
    l = Random(7, 28, 100)
    l.generate_sequences()
    d = PeptideDescriptor(l.sequences, 'eisenberg')
    d.calculate_moment()

    def test_filter_unnatural(self):
        self.r.filter_unnatural()
        self.assertNotIn('ABCDEFGHAJKLMNOPQRSTUVWXYZ', self.r.sequences)

    def test_mutate(self):
        self.r.mutate_AA(1, 1)
        self.assertNotEqual(self.sequences, self.r.sequences)

    def test_filter_aa(self):
        self.r.filter_aa(['G'])
        self.assertEqual(len(self.r.sequences), 0)

    def test_rand_selection(self):
        self.d.random_selection(10)
        self.assertEqual(len(self.d.sequences), 10)
        self.assertEqual(len(self.d.descriptor), 10)

if __name__ == '__main__':
    unittest.main()
