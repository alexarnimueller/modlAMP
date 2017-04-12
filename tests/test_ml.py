import unittest

from modlamp.datasets import load_ACPvsRandom
from modlamp.ml import train_best_model, score_cv
from modlamp.descriptors import PeptideDescriptor


class TestTrainModel(unittest.TestCase):

    data = load_ACPvsRandom()
    desc = PeptideDescriptor(data.sequences, scalename='pepcats')
    desc.calculate_autocorr(7)
    best_svm_model = train_best_model('svm', desc.descriptor, data.target, cv=5,
                                      param_grid=[{'clf__C': [1], 'clf__kernel': ['linear']}])
    best_rf_model = train_best_model('rf', desc.descriptor, data.target, cv=5,
                                     param_grid=[{'clf__n_estimators': [100]}])

    def test_score_svm(self):
        self.assertAlmostEqual(self.best_svm_model.score(self.desc.descriptor, self.data.target), 0.91767554, 3)

    def test_score_rf(self):
        self.assertAlmostEqual(self.best_rf_model.score(self.desc.descriptor, self.data.target), 1.0, 3)

    def test_score_cv(self):
        score = score_cv(self.best_svm_model, self.desc.descriptor, self.data.target, cv=10)
        self.assertAlmostEqual(score['mean'][0], 0.777, 3)


if __name__ == '__main__':
    unittest.main()
