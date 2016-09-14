import unittest

from ..ml import train_best_model
from ..datasets import load_ACPvsNeg
from ..descriptors import PeptideDescriptor

__author__ = 'modlab'

#
# class TestML(unittest.TestCase):
#
#     data = load_ACPvsNeg()
#     sequences = data.sequences
#     descr = PeptideDescriptor(sequences, 'pepcats')
#     descr.calculate_autocorr(7)
#     X_train = descr.descriptor
#     y_train = data.target
#     svm_params = {'C': 100.0,
#                  'cache_size': 200,
#                  'class_weight': None,
#                  'coef0': 0.0,
#                  'decision_function_shape': None,
#                  'degree': 3,
#                  'gamma': 0.001,
#                  'kernel': 'rbf',
#                  'max_iter': -1,
#                  'probability': True,
#                  'random_state': 1,
#                  'shrinking': True,
#                  'tol': 0.001,
#                  'verbose': False}
#
#     def test_train_best_model(self):
#         best_estim = train_best_model('svm', self.X_train, self.y_train)
#         self.assertEqual(best_estim.steps[1][1].get_params(), self.svm_params)
#         self.assertAlmostEqual(best_estim.score(self.X_train, self.y_train), 0.96825396825396826)
#
# if __name__ == '__main__':
#     unittest.main()
