# -*- coding: utf-8 -*-
"""Regression tests for descriptor state handling."""
import unittest

import numpy as np

from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

__author__ = "modlab"


class TestDescriptorState(unittest.TestCase):
    seqs = ["GLFDIVKKVVGALG", "KKLLKKLLKK"]

    def test_charge_density_append_keeps_previous_features(self):
        d = GlobalDescriptor(self.seqs)
        d.length()
        d.aromaticity(append=True)
        d.charge_density(append=True)
        self.assertEqual(d.featurenames, ["Length", "Aromaticity", "ChargeDensity"])
        self.assertEqual(d.descriptor.shape, (2, 3))

    def test_isoelectric_point_is_order_independent(self):
        forward = GlobalDescriptor(["DDDDEEEE", "KKKKKKKK"])
        forward.isoelectric_point()
        reverse = GlobalDescriptor(["KKKKKKKK", "DDDDEEEE"])
        reverse.isoelectric_point()
        self.assertAlmostEqual(forward.descriptor[0, 0], reverse.descriptor[1, 0], places=4)
        self.assertAlmostEqual(forward.descriptor[1, 0], reverse.descriptor[0, 0], places=4)

    def test_moment_rejects_multidimensional_scales(self):
        p = PeptideDescriptor(["GLFDIVKKVVGALGSL"], "pepcats")
        self.assertRaises(ValueError, p.calculate_moment)

    def test_all_moms_does_not_accumulate(self):
        q = PeptideDescriptor(["GLFDIVKKVVGALGSL", "KLLKLLKKLLKLLKKL"], "eisenberg")
        q.calculate_moment()
        q.calculate_moment()
        self.assertEqual(len(q.all_moms), 2)

    def test_profile_is_not_poisoned_by_an_earlier_call(self):
        a = PeptideDescriptor(["KLLKLLKKVVGALGGGLFDIVK"], "kytedoolittle")
        a.calculate_profile(prof_type="H", window=7)
        b = PeptideDescriptor(["KLLKLLKKVVGALGGGLFDIVK"], "kytedoolittle")
        b.calculate_global()
        b.calculate_profile(prof_type="H", window=7)
        np.testing.assert_allclose(a.descriptor, b.descriptor)

    def test_invalid_modality_raises(self):
        p = PeptideDescriptor(["GLFDIVKKVVGALGSL"], "eisenberg")
        self.assertRaises(ValueError, p.calculate_global, modality="maximum")
        self.assertRaises(ValueError, p.calculate_moment, modality="maximum")

    def test_arc_covers_the_c_terminal_window(self):
        # a 19-mer has two 18-residue windows; the old code evaluated only the first
        with_w = PeptideDescriptor(["GGGGGGGGGGGGGGGGGGW"], "peparc")
        with_w.calculate_arc()
        without_w = PeptideDescriptor(["GGGGGGGGGGGGGGGGGGG"], "peparc")
        without_w.calculate_arc()
        self.assertFalse(
            np.array_equal(with_w.descriptor, without_w.descriptor),
            "the C-terminal residue does not influence the arc descriptor",
        )


if __name__ == "__main__":
    unittest.main()
