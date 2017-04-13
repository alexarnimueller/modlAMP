import unittest
import numpy as np
from modlamp.ml import train_best_model, score_cv, score_testset


class TestTrainModel(unittest.TestCase):
    descriptor = np.array([[0.36581538], [0.17366842], [0.19780249], [0.56398772], [0.45888646], [0.37802301],
                           [0.34111542], [0.25833683], [0.29136177], [0.32289102], [0.10113402], [0.07952263],
                           [0.43523497], [0.27801969], [0.07287307], [0.27460903], [0.09064536], [0.13651603],
                           [0.30059792], [0.26340581]])
    descriptor_test = np.array([[0.68907562], [0.082342345]])
    
    target = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    target_test = np.array([1, 0])
    
    best_svm_model = train_best_model('svm', descriptor, target, cv=2, n_jobs=1,
                                      param_grid=[{'clf__C': [1], 'clf__kernel': ['rbf']}])
    best_rf_model = train_best_model('rf', descriptor, target, cv=2, n_jobs=1,
                                     param_grid=[{'clf__n_estimators': [100]}])
    
    def test_score_svm(self):
        self.assertAlmostEqual(self.best_svm_model.score(self.descriptor, self.target), 0.7, 3)
    
    def test_score_rf(self):
        self.assertAlmostEqual(self.best_rf_model.score(self.descriptor, self.target), 1.0, 3)
    
    def test_score_cv(self):
        score = score_cv(self.best_svm_model, self.descriptor, self.target, cv=2)
        self.assertAlmostEqual(score['mean'][1], 0.5, 3)
    
    def test_score_testset(self):
        score = score_testset(self.best_rf_model, self.descriptor_test, self.target_test)
        self.assertEqual(score['Scores'][0], 1.0)

if __name__ == '__main__':
    unittest.main()
