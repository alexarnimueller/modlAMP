README
======

**modlamp**

This is a python package that is designed for working with peptides, proteins or any amino acid sequence of natural amino acids. 
It incorporates several modules, like descriptor calculation (module **descriptors**) or sequence generation (module **sequences**).
For basic instructions how to use the package, see Usage_.


Installation
************

For the installation to work, ``pip`` needs to be installed. If you're not sure whether you already have pip, type
``pip --version``. If you don't have pip installed, install it via ``easy_install pip``.

When pip is installed, run the following command when located in the modlAMP package directory::

    make
    sudo make install
    make doc

Usage
*****

After installation, you should be able to import and use the different modules like shown below:

>>> from modlamp.sequences import Centrosymmetric
>>> from modlamp.descriptors import PeptideDescriptor
>>> S = Centrosymmetric(5)
>>> S.generate_asymmetric() # generate 5 asymmetric centrosymmetric sequences
>>> S.sequences
['VRVFVRVVKLYLKVVKGFGKV','ARAFARAVRGYGRV','VKIWIKVGRAFARG','LRIWIRLIKAYAKI','VKAYAKVVRLWLRV']

>>> D = PeptideDescriptor(S.sequences,'PPCALI') # describe the sequences by the auto-correlated PPCALI descriptor scale
>>> D.calculate_autocorr(7)
>>> D.save_descriptor('/Users/name/Desktop/descriptor.csv',delimiter=',')

A basic workflow how the package can be used is shown hereafter:

>>> from modlamp.sequences import MixedLibrary
>>> from modlamp.descriptors import PeptideDescriptor
>>> Lib = MixedLibrary(1000)
>>> Lib.generate_library()
>>> Lib.sequences[:10]
['VIVRVLKIAA','VGAKALRGIGPVVK','QTGKAKIKLVKLRAGPYANGKLF','RLIKGALKLVRIVGPGLRVIVRGAR','DGQTNRFCGI','ILRVGKLAAKV',...]

These commands generated a mixed peptide library comprising of 1000 sequences.

.. note::
    If duplicates are present in :py:att:`self.sequences`, these are removed during generation. Therefore it is possible
    that less than the specified sequences are obtained.

Now, different descriptor values can be calculated for these sequences:

>>> from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
>>> P = PeptideDescriptor(Lib.sequences,'eisenberg')
>>> P.calculate_moment()
>>> P.descriptor[:10]
array([[ 0.60138255],[ 0.61232763],[ 0.01474009],[ 0.72333858],[ 0.20390763],[ 0.68818279],...]

We calculated the global hydrophobic moments from the Eisenberg hydrophobicity scale. We can now plot these values
as a boxplot:

>>> from modlamp.plot import plot_feature
>>> plot_feature(P.descriptor,y_label='uH Eisenberg')

.. image:: ../../static/uH_boxplot.png