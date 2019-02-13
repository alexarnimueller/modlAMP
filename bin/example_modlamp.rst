Example scripts
===============

For documentation of the used modules see the the `Documentation <modlamp.html>`_ section.

A modlamp example script for peptide classification with a Random Forest classifier.
------------------------------------------------------------------------------------

.. code-block:: python

    import pandas as pd
    from modlamp.datasets import load_AMPvsUniProt
    from modlamp.descriptors import PeptideDescriptor
    from modlamp.ml import train_best_model, score_cv
    from modlamp.descriptors import PeptideDescriptor
    from modlamp.sequences import MixedLibrary

    # define the size of the peptide library to screen
    libsize = 1000
    print("Chosen library size is %i" % libsize)

    # load training sequences
    data = load_AMPvsUniProt()

    # describe sequences with PepCATS descriptor
    descr = PeptideDescriptor(data.sequences, 'pepcats')
    descr.calculate_crosscorr(7)

    # find best Random Forest classifier based on the PEPCATS data
    best_RF = train_best_model('RF', descr.descriptor, data.target)  # might take a while

    # evaluate performance of best model in 10-fold cross validation
    score_cv(best_RF, descr.descriptor, data.target, cv=10)

    # generate a virtual peptide library of `libsize` sequences to screen
    lib = MixedLibrary(libsize)
    lib.generate_sequences()
    print("Actual lirutal library size (without duplicates): %i" % len(lib.sequences))

    # describe library with PEPCATS descriptor
    lib_desc = PeptideDescriptor(lib.sequences, 'pepcats')
    lib_desc.calculate_crosscorr(7)

    # predict class probabilities for sequences in Library
    proba = best_RF.predict_proba(lib_desc.descriptor)

    # create ordered dictionary with sequences and prediction values and order it according to AMP predictions
    d = pd.DataFrame({'sequence': lib.sequences, 'prediction': proba[:, 1]})
    d50 = d.sort_values('prediction', ascending=False)[:50]  # 50 top AMP predictions

    # print the 50 top ranked predictions with their predicted probabilities
    print(d50)


An alternative example for single peptide classification with a RF classifier.
------------------------------------------------------------------------------
.. code-block:: python

    import pandas as pd
    from modlamp.datasets import load_AMPvsUniProt
    from modlamp.descriptors import PeptideDescriptor
    from modlamp.ml import train_best_model, score_cv
    from modlamp.descriptors import PeptideDescriptor

    ### IN THE FOLLOWING LIST, ADD YOUR SEQUENCES TO BE PREDICTED ###
    to_predict = ['GLLDSLLALLFEWASQ', 'KLLKLLKLLKLLKLLKKKLKLKL', 'GLFDDSKALLKKDFWWW']

    # load training sequences
    data = load_AMPvsUniProt()

    # describe sequences with PepCATS descriptor
    descr = PeptideDescriptor(data.sequences, 'pepcats')
    descr.calculate_crosscorr(7)

    # train a Random Forest classifier with given parameters based on the PEPCATS data
    best_RF = train_best_model('RF', descr.descriptor, data.target, cv=2,
                                param_grid={'clf__bootstrap': [True], 'clf__criterion': ['gini'], 'clf__max_features':
                                ['sqrt'], 'clf__n_estimators': [500]})

    # evaluate performance of the model in 5-fold cross validation
    score_cv(best_RF, descr.descriptor, data.target, cv=5)

    # describe sequences to be predicted with PEPCATS descriptor
    lib_desc = PeptideDescriptor(to_predict, 'pepcats')
    lib_desc.calculate_crosscorr(7)

    # predict class probabilities for the desired sequences
    proba = best_RF.predict_proba(lib_desc.descriptor)

    # create ordered dictionary with sequences and prediction values and order it according to AMP predictions
    d = pd.DataFrame({'sequence': to_predict, 'prediction': proba[:, 1]})
    d = d.sort_values('prediction', ascending=False)
    print(d)  # print the final predictions (sorted according to decreasing probabilities)


Loading sequences from a ``FASTA`` file
---------------------------------------

A further example of how to load a list of own amino acid sequences from a ``.FASTA`` formatted file, calculate
descriptors and save the values back to a ``.csv`` file.

.. code-block:: python

    from modlamp.descriptors import PeptideDescriptor

    # load sequences from FASTA file and calculate the pepcats cross-correlated descriptor
    x = PeptideDescriptor('location/of/your/file.fasta', 'pepcats')
    x.calculate_crosscorr(window=7)
    # save calculated descriptor to a .csv file
    x.save_descriptor('location/of/your/outputfile.csv', delimiter=',')


Combining different descriptors & saving to ``csv``
---------------------------------------------------

Many more descriptors are available for calculations. Here is another example of reading a sequence file and
calculating two sets of descriptors followed by saving them to ``.csv`` files.

.. code-block:: python

    from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

    # Load sequence file into descriptor object
    pepdesc = PeptideDescriptor('/path/to/sequences.fasta', 'eisenberg')  # use Eisenberg consensus scale
    globdesc = GlobalDescriptor('/path/to/sequences.fasta')

    # --------------- Peptide Descriptor (AA scales) Calculations ---------------
    pepdesc.calculate_global()  # calculate global Eisenberg hydrophobicity
    pepdesc.calculate_moment(append=True)  # calculate Eisenberg hydrophobic moment

    # load other AA scales
    pepdesc.load_scale('gravy')  # load GRAVY scale
    pepdesc.calculate_global(append=True)  # calculate global GRAVY hydrophobicity
    pepdesc.calculate_moment(append=True)  # calculate GRAVY hydrophobic moment
    pepdesc.load_scale('z3')  # load old Z scale
    pepdesc.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)

    # save descriptor data to .csv file
    col_names1 = 'ID,Sequence,H_Eisenberg,uH_Eisenberg,H_GRAVY,uH_GRAVY,Z3_1,Z3_2,Z3_3'
    pepdesc.save_descriptor('/path/to/descriptors1.csv', header=col_names1)

    # --------------- Global Descriptor Calculations ---------------
    globdesc.length()  # sequence length
    globdesc.boman_index(append=True)  # Boman index
    globdesc.aromaticity(append=True)  # global aromaticity
    globdesc.aliphatic_index(append=True)  # aliphatic index
    globdesc.instability_index(append=True)  # instability index
    globdesc.calculate_charge(ph=7.4, amide=False, append=True)  # net charge
    globdesc.calculate_MW(amide=False, append=True)  # molecular weight

    # save descriptor data to .csv file
    col_names2 = 'ID,Sequence,Length,BomanIndex,Aromaticity,AliphaticIndex,InstabilityIndex,Charge,MW'
    globdesc.save_descriptor('/path/to/descriptors2.csv', header=col_names2)
