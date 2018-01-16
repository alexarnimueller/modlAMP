import unittest

import numpy as np
from os.path import dirname, join
from modlamp.plot import plot_violin, plot_2_features, plot_3_features, plot_aa_distr, plot_feature, plot_profile, plot_pde


class TestPlots(unittest.TestCase):
    data = np.random.random((3, 100))
    sequences = ['AAAACCCCDDDDDEEELLLSSSKKKKKALLAKLSKEEEIIQIIWWWWWPLRTLLNS', 'KLLKLLKVVGALGWI', 'KKKKKKK', 'RRRRR']
    targets = np.random.randint(0, 2, 100)
    fname = join(dirname(__file__), 'files/plots/testplot.png')

    def test_feature(self):
        plot_feature(self.data[0], x_tick_labels=['1', '0'], targets=self.targets, filename=self.fname)

    def test_violin(self):
        plot_violin(self.data[0], filename=self.fname)

    def test_2features(self):
        plot_2_features(self.data[0], self.data[1], targets=self.targets, filename=self.fname)

    def test_3features(self):
        plot_3_features(self.data[0], self.data[1], self.data[2], targets=self.targets, filename=self.fname)

    def test_plot_aa(self):
        plot_aa_distr(self.sequences, filename=self.fname)

    def test_profile(self):
        plot_profile(self.sequences[1], filename=self.fname)
        self.assertRaises(KeyError, plot_profile, 'GLFDIVKKVVLVLVLV', 5, 'pepcats', self.fname)

    def test_pde(self):
        plot_pde(self.data, filename=self.fname)


if __name__ == '__main__':
    unittest.main()
