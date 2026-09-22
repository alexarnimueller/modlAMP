# -*- coding: utf-8 -*-
"""Regression tests for the reported cross-validation metrics."""
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from modlamp.ml import score_cv, score_testset

__author__ = "modlab"


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(0)
        cls.x = np.vstack([rng.normal(0, 1, (40, 4)), rng.normal(1.5, 1, (40, 4))])
        cls.y = np.array([0] * 40 + [1] * 40)
        cls.clf = Pipeline([("scl", StandardScaler()), ("clf", SVC(probability=True, random_state=1))])
        cls.clf.fit(cls.x, cls.y)

    def test_std_is_computed_over_folds_only(self):
        df = score_cv(self.clf, self.x, self.y, cv=4)
        folds = [c for c in df.columns if c.startswith("CV_")]
        np.testing.assert_allclose(df["std"].values, df[folds].std(axis=1).round(2).values, atol=0.011)

    def test_cv_roc_auc_uses_continuous_scores(self):
        df = score_cv(self.clf, self.x, self.y, cv=4)
        label_auc = round(roc_auc_score(self.y, self.clf.predict(self.x)), 2)
        self.assertNotAlmostEqual(df.loc["roc_auc", "mean"], label_auc, places=3)

    def test_testset_roc_auc_matches_probability_auc(self):
        df = score_testset(self.clf, self.x, self.y)
        expected = round(roc_auc_score(self.y, self.clf.predict_proba(self.x)[:, 1]), 2)
        self.assertAlmostEqual(df.loc["roc_auc", "Scores"], expected, places=2)

    def test_all_metric_rows_are_present(self):
        df = score_cv(self.clf, self.x, self.y, cv=4)
        for m in ["MCC", "accuracy", "precision", "recall", "f1", "roc_auc",
                  "TN", "FP", "FN", "TP", "FDR", "sensitivity", "specificity"]:
            self.assertIn(m, df.index)


if __name__ == "__main__":
    unittest.main()
