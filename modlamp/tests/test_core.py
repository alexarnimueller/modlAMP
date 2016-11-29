import unittest
from modlamp.core import BaseSequence
from modlamp.sequences import Random
from modlamp.descriptors import PeptideDescriptor


class TestCore(unittest.TestCase):
    b = BaseSequence(1, 10, 20)
    b.sequences = ['GLFDIVKKVVGALG', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALK', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'AGGURST', 'aggo']
    s = PeptideDescriptor(['GLFDIVKKVVGALG', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALK', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'AGGURST', 'aggorst'])
    l = Random(100, 7, 28)
    l.generate_sequences()
    d = PeptideDescriptor(l.sequences, 'eisenberg')
    d.calculate_moment()

    def test_filter_aa(self):
        self.b.filter_aa(['C'])
        self.assertEqual(len(self.b.sequences), 5)

    def test_filter_duplicates(self):
        self.b.filter_duplicates()
        self.assertEqual(len(self.b.sequences), 4)

    def test_keep_natural_aa(self):
        self.s.keep_natural_aa()
        self.assertNotIn(['ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'AGGURST', 'aggorst'], self.s.sequences)

    def test_mutate(self):
        self.b.mutate_AA(1, 1.)
        self.assertNotEqual('GLFDIVKKVVGALG', self.b.sequences[0])

    def test_rand_selection(self):
        self.d.random_selection(10)
        self.assertEqual(len(self.d.sequences), 10)
        self.assertEqual(len(self.d.descriptor), 10)

if __name__ == '__main__':
    unittest.main()
