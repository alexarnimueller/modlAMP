# -*- coding: utf-8 -*-
"""Regression tests for scikit-learn / matplotlib API compatibility."""
import os
import tempfile
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from modlamp.ml import (  # noqa: E402
    plot_validation_curve,
    predict,
    score_cv,
    train_best_model,
)
from modlamp.plot import plot_2_features, plot_3_features, plot_feature  # noqa: E402

__author__ = "modlab"


class TestSklearnCompat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(0)
        cls.x = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(1.5, 1, (30, 4))])
        cls.y = np.array([0] * 30 + [1] * 30)
        cls.clf = Pipeline([("scl", StandardScaler()), ("clf", SVC(probability=True, random_state=1))])
        cls.clf.fit(cls.x, cls.y)

    def _tmp(self, suffix):
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self.addCleanup(os.remove, path)
        return path

    def test_score_cv_without_shuffle(self):
        df = score_cv(self.clf, self.x, self.y, cv=4, shuffle=False)
        self.assertIn("mean", df.columns)

    def test_predict_accepts_array_targets_and_names(self):
        df = predict(self.clf, self.x[:3], seqs=["AAA", "CCC", "DDD"], names=["a", "b", "c"], y=self.y[:3])
        self.assertIn("True_class", df.columns)
        self.assertIn("Name", df.columns)

    def test_train_best_model_accepts_sample_weights(self):
        m = train_best_model(
            "svm",
            self.x,
            self.y,
            sample_weights=np.ones(60),
            cv=3,
            param_grid=[{"clf__C": [1.0], "clf__kernel": ["linear"]}],
        )
        self.assertIsNotNone(m)

    def test_validation_curve_runs(self):
        path = self._tmp(".png")
        plot_validation_curve(
            self.clf, self.x, self.y, param_name="clf__C", param_range=[0.1, 1.0], cv=3, filename=path
        )
        self.assertGreater(os.path.getsize(path), 0)


class TestMatplotlibCompat(unittest.TestCase):
    data = np.random.RandomState(1).rand(3, 20)
    targets = np.array([0] * 10 + [1] * 10)

    def _tmp(self):
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        self.addCleanup(os.remove, path)
        return path

    def test_plot_feature(self):
        path = self._tmp()
        plot_feature(self.data[0], targets=self.targets, filename=path)
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_2_features(self):
        path = self._tmp()
        plot_2_features(self.data[0], self.data[1], targets=self.targets, filename=path)
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_3_features(self):
        path = self._tmp()
        plot_3_features(self.data[0], self.data[1], self.data[2], targets=self.targets, filename=path)
        self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
