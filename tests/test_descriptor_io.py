# -*- coding: utf-8 -*-
"""Regression tests for descriptor import/export."""
import os
import tempfile
import unittest

import numpy as np

from modlamp.descriptors import GlobalDescriptor

__author__ = "modlab"


class TestDescriptorIO(unittest.TestCase):
    def _tmp(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        self.addCleanup(os.remove, path)
        return path

    def test_header_is_not_truncated_to_first_letters(self):
        d = GlobalDescriptor(["GLFDIVKKVVGALG", "KKLLKKLLKK"])
        d.calculate_all()
        path = self._tmp()
        d.save_descriptor(path)
        with open(path) as f:
            header = f.readline().strip().split(",")
        self.assertEqual(header, ["Sequence"] + d.featurenames)

    def test_header_names_the_target_column(self):
        d = GlobalDescriptor(["GLFDIVKKVVGALG", "KKLLKKLLKK"])
        d.calculate_all()
        path = self._tmp()
        d.save_descriptor(path, targets=np.array([0, 1]))
        with open(path) as f:
            header = f.readline().strip().split(",")
        self.assertEqual(header[-1], "Target")

    def test_load_descriptordata_excludes_the_target_column(self):
        d = GlobalDescriptor(["GLFDIVKKVVGALG", "KKLLKKLLKK"])
        d.calculate_all()
        n_features = d.descriptor.shape[1]
        path = self._tmp()
        d.save_descriptor(path, targets=np.array([0, 1]))
        loaded = GlobalDescriptor(["GLFDIVKKVVGALG"])
        loaded.load_descriptordata(path, targets=True, skip_header=1)
        self.assertEqual(loaded.descriptor.shape[1], n_features)
        np.testing.assert_array_equal(loaded.target, np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
