README
======

**modlamp**

This is a python package that is designed for working with peptides, proteins or any amino acid sequence of natural amino acids. 
It incorporates several modules, like descriptor calculation (module **descriptors**) or sequence generation (module **sequences**).
For basic instructions how to use the package, see Usage_ or this `example script <examplescriptinclude.html>`_.


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

Importing modules
-----------------

After installation, you should be able to import and use the different modules like shown below:

>>> from modlamp.sequences import Centrosymmetric
>>> from modlamp.descriptors import PeptideDescriptor
>>> from modlamp.database import query_sequences

Generating Sequences
--------------------

The following example shows how to generate a library of 1000 sequences out of all available sequence generation methods:

>>> from modlamp.sequences import MixedLibrary
>>> Lib = MixedLibrary(1000)
>>> Lib.generate_library()
>>> Lib.sequences[:10]
['VIVRVLKIAA','VGAKALRGIGPVVK','QTGKAKIKLVKLRAGPYANGKLF','RLIKGALKLVRIVGPGLRVIVRGAR','DGQTNRFCGI','ILRVGKLAAKV',...]

These commands generated a mixed peptide library comprising of 1000 sequences.

.. note::
    If duplicates are present in :py:attr:`self.sequences`, these are removed during generation. Therefore it is possible
    that less than the specified sequences are obtained.

The module :mod:`sequences` incorporates different sequence generation classes. For documentation thereof, consider the
docs for `sequences <modlamp.html#module-modlamp.sequences>`_.

Calculating Descriptor Values
-----------------------------

Now, different descriptor values can be calculated for the generated sequences: (see `Generating Sequences`_)

>>> from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
>>> P = PeptideDescriptor(Lib.sequences,'eisenberg')
>>> P.calculate_moment()
>>> P.descriptor[:10]
array([[ 0.60138255],[ 0.61232763],[ 0.01474009],[ 0.72333858],[ 0.20390763],[ 0.68818279],...]
>>> G = GlobalDescriptor(Lib.sequences)
>>> G.isoelectric_point()
>>> G.descriptor[:10]
array([ 10.09735107,   8.75006104,  12.30743408,  11.26385498, ...]

We calculated the global hydrophobic moments from the Eisenberg hydrophobicity scale and the isoelectric points.
Many more descriptors can be calculated, from global descriptors to concoluted / correlated descriptors from different
amino acid scales. For further information consider the docs for `descriptors <modlamp.html#module-modlamp.descriptors>`_.

Plotting Features
-----------------

We can now plot these values as a boxplot, for example the hydrophobic moment:

>>> from modlamp.plot import plot_feature
>>> plot_feature(P.descriptor,y_label='uH Eisenberg')

.. image:: static/uH_Eisenberg.png

We can additionally compare these descriptor values to known AMP sequences. For that, we import sequences from the APD3, which
are stored in the FASTA formatted file ``APD3.fasta``:

>>> APD = PeptideDescriptor('/Path/to/file/APD3.fasta','eisenberg')
>>> APD.calculate_moment()

Now lets compare the values by plotting:

>>> plot_feature((P.descriptor,APD.descriptor),y_label='uH Eisenberg',x_tick_labels=['Library','APD3'])

.. image:: static/uH_APD3.png

It is also possible to plot 2 or 3 different features in a scatter plot:

>>> from modlamp.plot import plot_3_features
>>> A = PeptideDescriptor('/Path/to/file/APD3.fasta','eisenberg')
>>> A.calculate_moment()
>>> B = GlobalDescriptor('/Path/to/file/APD3.fasta')
>>> B.isoelectric_point()
>>> C = GlobalDescriptor('/Path/to/file/APD3.fasta')
>>> C.length()
>>> plot_3_features(A.descriptor,B.descriptor,C.descriptor,x_label='uH',y_label='pI',z_label='length')

.. image:: static/3D_scatter.png

Database Connection
-------------------

modlamp hosts a module for connecting to the modlab internal peptide database on the gsdelta641 server.
Peptide sequences included in any table in the peptides database can be downloaded directly in python.

.. warning::
    This module only works in the modlab intranet at ETH Zurich

For querying sequences from a given table, the sequences must be stored in a column called "sequences" in the mysql table.
The query then works as follows:

>>> from modlamp.database import query_sequences
>>> query_sequences('modlab_experiments')
Password: >? ***********
Connecting to MySQL database...
connection established!
['ILGTILGILKGL','ILGTILGFLKGL','ILGNILGFLKGL','ILGQILGILKGL','ILGHILGYLKGL','PAGHILGWWKGL','GLFDIVKKVVGALG',...]
