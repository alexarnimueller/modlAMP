import unittest
from modlamp.core import BaseSequence, BaseDescriptor
from modlamp.sequences import Random
from modlamp.descriptors import PeptideDescriptor
from os.path import dirname, join


class TestCore(unittest.TestCase):
    b = BaseSequence(1, 10, 20)
    b.sequences = ['GLFDIVKKVVGALG', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALK', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'AGGURST',
                   'aggo']
    n = BaseDescriptor('GLFDIVKKVVGALGSLGLFDIVKKVVGALGSL')
    b.names = ['1', '2', '3', '4', '5', '6']
    s = PeptideDescriptor(
        ['GLFDIVKKVVGALG', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALK', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'AGGURST', 'aggorst'])
    s.names = b.names
    l = Random(100, 7, 28)
    l.generate_sequences()
    d = PeptideDescriptor(l.sequences, 'eisenberg')
    d.calculate_moment()

    def test_ngrams(self):
        self.n.count_ngrams([2, 3])
        self.assertEqual(self.n.descriptor['ALG'], 2)

    def test_filter_aa(self):
        self.b.filter_aa(['C'])
        self.assertEqual(len(self.b.sequences), 5)

    def test_filter_duplicates(self):
        self.b.filter_duplicates()
        self.assertEqual(len(self.b.sequences), 4)

    def test_keep_natural_aa(self):
        self.assertIn('ABCDEFGHIJKLMNOPQRSTUVWXYZ', self.s.sequences)
        self.s.keep_natural_aa()
        self.assertNotIn('ABCDEFGHIJKLMNOPQRSTUVWXYZ', self.s.sequences)

    def test_mutate(self):
        self.b.mutate_AA(2, 1.)
        self.assertNotEqual('GLFDIVKKVVGALG', self.b.sequences[0])

    def test_rand_selection(self):
        self.d.random_selection(10)
        self.assertEqual(len(self.d.sequences), 10)
        self.assertEqual(len(self.d.descriptor), 10)

    def test_safe_fasta(self):
        self.d.save_fasta(join(dirname(__file__), 'files/saved.fasta'), names=True)
        self.d.save_fasta(join(dirname(__file__), 'files/saved.fasta'), names=False)


if __name__ == '__main__':
    unittest.main()
