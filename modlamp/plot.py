"""
.. module:: modlamp.plot

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to plot different feature plots. The following functions are available:

============================		============================================================================
Function							Characteristics
============================		============================================================================
:py:func:`plot_feature`				Generate a box plot for visualizing the distribution of a given feature.
:py:func:`plot_2_features`			Generate a 2D scatter plot of 2 given features.
:py:func:`plot_3_features`			Generate a 3D scatter plot of 3 given features.
============================		============================================================================

"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

__author__ = "modlab"
__docformat__ = "restructuredtext en"


def plot_feature(y_values, targets=None, y_label='feature values', x_tick_labels=None, filename=None):
	"""
	Function to generate a box plot of 1 given feature. The different target classes given in **targets** are plottet as separate boxes.

	:param y_values: Array of feature values to be plotted.
	:param targets: List of target class values [string/binary] for the given feature data.
	:param y_label: Axis label.
	:param x_tick_labels: list of labels to be assigned to the ticks on the x-axis. Must match the number of targets.
	:param filename: filename where to safe the plot. *default = None*
	:return: A feature box plot.
	:Example:

	>>> plot_feature(P.descriptor,y_label='uH Eisenberg')

	.. image:: ../docs/static/uH_Eisenberg.png
		:scale: 50 %
	"""
	colors = ['dodgerblue','firebrick','gold','lightgreen','navy','black','hotpink'] # available colors

	fig, ax = plt.subplots()

	if targets:
		data = []
		cntr = 0
		for n in list(set(targets)): # finding indices of the different targets in "targets" and plotting
			t = np.array([i for i,j in enumerate(targets) if j == n])
			data.append([y_values[t]])
			cntr += 1
		if x_tick_labels:
			labels = x_tick_labels
		else:
			labels = range(cntr)
		colors = colors[:cntr]

	else:
		if x_tick_labels:
			labels = x_tick_labels
		else:
			labels = ['all data']
		data = y_values

	# coloring faces of boxes
	medianprops = dict(linestyle='-', linewidth=1, color='black')
	box = ax.boxplot(data,notch=True, patch_artist=True,medianprops=medianprops,labels=labels)
	plt.setp(box['whiskers'], color='black')

	for patch, color in zip(box['boxes'], colors):
		patch.set(facecolor=color,edgecolor='black',alpha=0.8)

	ax.set_xlabel('Classes',fontweight='bold')
	ax.set_ylabel(y_label, fontweight='bold')
	ax.set_title('Feature Box-Plot',fontsize=16,fontweight='bold')

	if filename:
		plt.savefig(filename,dpi=150)
	else:
		plt.show()


def plot_2_features(x_values, y_values, targets=None, x_label='', y_label='', filename=None):
	"""
	Function to generate a feature scatter plot of 2 given features. The different target classes given in **targets**
	are plottet in different colors.

	:param x_values: Array of values of the feature to be plotted on the x-axis.
	:param y_values: Array of values of the feature to be plotted on the y-axis.
	:param targets: List of target class values [string/binary] for the given feature data.
	:param x_label: X-axis label.
	:param y_label: Y-axis label.
	:param filename: filename where to safe the plot. *default = None*
	:return: A 2D feature scatter plot.
	:Example:

	>>> plot_2_features(A.descriptor,B.descriptor,x_label='uH',y_label='pI',targets=targets)

	.. image:: ../docs/static/2D_scatter.png
		:scale: 50 %
	"""

	colors = ['dodgerblue','firebrick','gold','lightgreen','navy','black'] # available colors

	fig, ax = plt.subplots()

	if targets:
		for n in list(set(targets)):  # finding indices of the different targets in "targets" and plotting
			t = np.array([i for i,j in enumerate(targets) if j == n])
			xt = x_values[t]  # find all values in x for the given target
			yt = y_values[t]  # find all values in y for the given target
			ax.scatter(xt, yt, c=colors[n], alpha=1., s=25, label='class '+str(n))  # plot scatter for this target group
			ax.legend(loc='lower right')

	else:
		ax.scatter(x_values, y_values, c=colors[0], alpha=1., s=25)

	ax.set_xlabel(x_label, fontweight='bold')
	ax.set_ylabel(y_label, fontweight='bold')
	ax.set_title('2D Feature Plot', fontsize=16, fontweight='bold')

	if filename:
		plt.savefig(filename, dpi=150)
	else:
		plt.show()


def plot_3_features(x_values, y_values, z_values, targets=None, x_label='', y_label='', z_label='', filename=None):
	"""
	Function to generate a 3D feature scatter plot of 3 given features. The different target classes given in **targets**
	are plottet in different colors.

	:param x_values: Array of values of the feature to be plotted on the x-axis.
	:param y_values: Array of values of the feature to be plotted on the y-axis.
	:param z_values: Array of values of the feature to be plotted on the z-axis.
	:param targets: List of target class values {string/binary} for the given feature data.
	:param x_label: {str} X-axis label.
	:param y_label: {str} Y-axis label.
	:param z_label: {str} Z-axis label.
	:param filename: {str} filename where to safe the plot. *default = None* -> show the plot
	:return: A 3D feature scatter plot.
	:Example:

	>>> plot_3_features(A.descriptor,B.descriptor,C.descriptor,x_label='uH',y_label='pI',z_label='length')

	.. image:: ../docs/static/3D_scatter.png
		:scale: 50 %
	"""

	colors = ['dodgerblue', 'firebrick', 'gold', 'lightgreen', 'navy', 'black']  # available colors

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	if targets:
		for n in list(set(targets)):  # finding indices of the different targets in "targets" and plotting
			t = np.array([i for i, j in enumerate(targets) if j == n])
			xt = x_values[t]  # find all values in x for the given target
			yt = y_values[t]  # find all values in y for the given target
			zt = z_values[t]  # find all values in y for the given target
			ax.scatter(xt, yt, zt, c=colors[n], alpha=1., s=25, label='class '+str(n))  # plot 3Dscatter for this target
			ax.legend(loc='best')

	else:  # plot 3Dscatter for this target group
		ax.scatter(x_values, y_values, z_values, c=colors[0], alpha=1., s=25)

	ax.set_xlabel(x_label, fontweight='bold')
	ax.set_ylabel(y_label, fontweight='bold')
	ax.set_zlabel(z_label, fontweight='bold')
	ax.set_title('3D Feature Plot', fontsize=16, fontweight='bold')

	if filename:
		plt.savefig(filename, dpi=150)
	else:
		plt.show()


def plot_profile(sequence, window=5, scalename='eisenberg', filename=None):
	""" Function to generate sequence profile plots of a given amino acid scale or a moment thereof.

	.. note::
		:func`plot_profile` can only plot one-dimensional amino acid scales given in :class:`PeptideDescriptor`.

	:param sequence: {str} Peptide sequence for which the profile should be plotted.
	:param window: {int, uneven} Window size for which the average value is plotted for the center amino acid.
	:param scalename: {str} Amino acid scale to be used to describe the sequence.
	:param filename: {str} Filename  where to safe the plot. *default = None* -> show the plot
	:return: a profile plot of the input sequence interactively or with the specified *filename*
	:Example:

	>>> plot_profile('GLFDIVKKVVGALGSL', scalename='eisenberg')

	.. image:: ../docs/static/profileplot.png
		:scale: 50 %
	"""
	# check if given scale is defined in PeptideDescriptor
	try:
		D = PeptideDescriptor(sequence, scalename)
	except TypeError:
		print("\nError\nNo sequence given!")
	except KeyError:
		print("\nSorry\nThis function cannot calculate a profile for the given scale '%s'." % scalename)
		print("Use the one dimensional scales given in the documentation for modlamp.descriptors.PeptideDescriptors")
	else:
		seq_data = list()
		seq_profile = list()
		for a in sequence:
			seq_data.append(D.scale[a])  # describe sequence by given scale
		i = 0  # AA index
		while (i + window) < len(sequence):
			seq_profile.append(np.mean(seq_data[i:(i + window + 1)]))  # append average value for given window
			i += 1

		# plot
		fig, ax = plt.subplots()
		x_range = range(int(window)/2 + 1, len(sequence)-int(window)/2)
		line = ax.plot(x_range, seq_profile)
		plt.setp(line, color='red', linewidth=2.0)

		# axis labes and title
		ax.set_xlabel('amino acid position', fontweight='bold')
		ax.set_ylabel(scalename + ' value', fontweight='bold')
		ax.set_title('Sequence Profile For ' + sequence, fontsize=16, fontweight='bold')
		ax.text(max(x_range)/2 + 1, 1.05 * max(seq_profile), 'window size: ' + str(window), fontsize=12)

		# only left and bottom axes, no box
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')

		# show or save plot
		if filename:
			plt.savefig(filename, dpi=150)
		else:
			plt.show()

# TODO: helical wheel plot function?
