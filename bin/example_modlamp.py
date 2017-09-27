"""
An example script for a possible usage of the modlamp package for peptide QSAR studies with a Random Forest model.
Give the number sequences to be generated as sys.argv[1]

:Example:
python example_modlamp.py 1000
"""

import sys
from collections import OrderedDict
from modlamp.datasets import load_AMPvsUniProt
from modlamp.descriptors import PeptideDescriptor
from sklearn.ensemble import RandomForestClassifier
from modlamp.sequences import MixedLibrary


def main(libsize=1000):
    # load training sequences
    data = load_AMPvsUniProt()

    # describe sequences with PEPCATS descriptor
    X = PeptideDescriptor(data.sequences, 'pepcats')
    X.calculate_crosscorr(7)
    
    # initialize Random Forest classifier
    clf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1)
    
    # fit the classifier on the PEPCATS data
    clf.fit(X.descriptor, data.target)
    
    # evaluate classifier performance as RF out of bag score
    print("RandomForest OOB classifcation score: %.3f" % clf.oob_score_)
    
    # generate a virtual peptide library of `size` sequences to screen
    Lib = MixedLibrary(libsize)
    Lib.generate_sequences()
    print("Actual lirutal library size (without duplicates): %i" % len(Lib.sequences))
    
    # describe library with PEPCATS descriptor
    X_lib = PeptideDescriptor(Lib.sequences, 'pepcats')
    X_lib.calculate_crosscorr(7)
    
    # predict class probabilities for sequences in Library
    proba = clf.predict_proba(X_lib.descriptor)
    
    # create ordered dictionary with sequences and prediction values and order it according to AMP predictions
    d = dict(zip(Lib.sequences, proba[:, 1]))
    d50 = OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True)[:50])  # 50 top AMP predictions
    
    # print the 50 top ranked predictions with their predicted probabilities
    print("Sequence,Predicted_AMP_Probability")
    for k in d50.keys():
        print k + "," + str(d50[k])


if __name__ == "__main__":
    main(sys.argv[1])
