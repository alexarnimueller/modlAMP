modlAMP example script
======================

A modlamp example script for peptide classification with help of a Random Forest classifier.


For documentation of the used modules see the the `Documentation <modlamp.html>`_ section.

.. code-block:: python

    import pandas as pd
    from modlamp.datasets import load_helicalAMPset
    from modlamp.descriptors import PeptideDescriptor
    from sklearn.ensemble import RandomForestClassifier
    from modlamp.sequences import MixedLibrary

    # define the size of the peptide library to screen
    libsize = 1000
    print("Chosen library size is %i" % libsize)

    # load training sequences
    data = load_helicalAMPset()

    # describe sequences with PEPCATS descriptor
    X = PeptideDescriptor(data.sequences, 'pepcats')
    X.calculate_crosscorr(7)

    # initialize Random Forest classifier
    clf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=1)

    # fit the classifier on the PEPCATS data
    clf.fit(X.descriptor, data.target)

    # evaluate classifier performance as RF out of bag score
    print("RandomForest OOB classifcation score: %.3f" % clf.oob_score_)

    # generate a virtual peptide library of `libsize` sequences to screen
    Lib = MixedLibrary(libsize)
    Lib.generate_library()
    print("Actual lirutal library size (without duplicates): %i" % len(Lib.sequences))

    # describe library with PEPCATS descriptor
    X_lib = PeptideDescriptor(Lib.sequences, 'pepcats')
    X_lib.calculate_crosscorr(7)

    # predict class probabilities for sequences in Library
    proba = clf.predict_proba(X_lib.descriptor)

    # create ordered dictionary with sequences and prediction values and order it according to AMP predictions
    d = pd.DataFrame({'sequence': Lib.sequences, 'prediction': proba[:, 1]})
    d50 = d.sort_values('prediction', ascending=False)[:50]  # 50 top AMP predictions

    # print the 50 top ranked predictions with their predicted probabilities
    print d50

A further example of how to load a list of own amino acid sequences from a ``.FASTA`` formatted file, calculate
descriptors and save the values back to a ``.csv`` file.

.. code-block:: python

    from modlamp.descriptors import PeptideDescriptor

    # load sequences from FASTA file and calculate the pepcats cross-correlated descriptor
    x = PeptideDescriptor('Location/of/your/file.fasta', 'pepcats')
    x.calculate_crosscorr(window=7)
    # save calculated descriptor to a .csv file
    x.save_descriptor('Location/of/your/outputfile.csv', delimiter=',')
