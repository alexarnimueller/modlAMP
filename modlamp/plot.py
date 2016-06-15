"""
.. module:: modlamp.plot

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to plot different feature plots. The following functions are available:

============================		==============================================================================
Function							Characteristics
============================		==============================================================================
:py:func:`plot_feature`				Generate a box plot for visualizing the distribution of a given feature.
:py:func:`plot_2_features`			Generate a 2D scatter plot of 2 given features.
:py:func:`plot_3_features`			Generate a 3D scatter plot of 3 given features.
:py:func:`plot_profile`				Generates a profile plot of a sequence to visualize potential linear gradients
:py:func:`helical_wheel`			Generates a helical wheel projection plot of a given sequence.
============================		==============================================================================

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from mpl_toolkits.mplot3d import Axes3D
from modlamp.descriptors import PeptideDescriptor
import pandas as pd
from scipy.stats.kde import gaussian_kde

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
	colors = ['dodgerblue', 'firebrick', 'gold', 'lightgreen', 'navy', 'black', 'hotpink']  # available colors

	fig, ax = plt.subplots()

	if targets:
		data = []
		cntr = 0
		for n in list(set(targets)):  # finding indices of the different targets in "targets" and plotting
			t = np.array([i for i, j in enumerate(targets) if j == n])
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
	box = ax.boxplot(data, notch=True, patch_artist=True, medianprops=medianprops, labels=labels)
	plt.setp(box['whiskers'], color='black')

	for patch, color in zip(box['boxes'], colors):
		patch.set(facecolor=color, edgecolor='black', alpha=0.8)

	ax.set_xlabel('Classes', fontweight='bold')
	ax.set_ylabel(y_label, fontweight='bold')
	ax.set_title('Feature Box-Plot', fontsize=16, fontweight='bold')

	if filename:
		plt.savefig(filename, dpi=150)
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

	colors = ['dodgerblue', 'firebrick', 'gold', 'lightgreen', 'navy', 'black']  # available colors

	fig, ax = plt.subplots()

	if targets:
		for n in list(set(targets)):  # finding indices of the different targets in "targets" and plotting
			t = np.array([i for i, j in enumerate(targets) if j == n])
			xt = x_values[t]  # find all values in x for the given target
			yt = y_values[t]  # find all values in y for the given target
			ax.scatter(xt, yt, c=colors[n], alpha=1., s=25,
					   label='class ' + str(n))  # plot scatter for this target group
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
			ax.scatter(xt, yt, zt, c=colors[n], alpha=1., s=25,
					   label='class ' + str(n))  # plot 3Dscatter for this target
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
		:func:`plot_profile` can only plot one-dimensional amino acid scales given in
		:class:`modlamp.descriptors.PeptideDescriptor`.

	:param sequence: {str} Peptide sequence for which the profile should be plotted.
	:param window: {int, uneven} Window size for which the average value is plotted for the center amino acid.
	:param scalename: {str} Amino acid scale to be used to describe the sequence.
	:param filename: {str} Filename  where to safe the plot. *default = None* --> show the plot
	:return: a profile plot of the input sequence interactively or with the specified *filename*
	:Example:

	>>> plot_profile('GLFDIVKKVVGALGSL', scalename='eisenberg')

	.. image:: ../docs/static/profileplot.png
		:scale: 50 %

	.. versionadded:: v2.1.5
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
		x_range = range(int(window) / 2 + 1, len(sequence) - int(window) / 2)
		line = ax.plot(x_range, seq_profile)
		plt.setp(line, color='red', linewidth=2.0)

		# axis labes and title
		ax.set_xlabel('sequence position', fontweight='bold')
		ax.set_ylabel(scalename + ' value', fontweight='bold')
		ax.set_title('Sequence Profile For ' + sequence, fontsize=16, fontweight='bold')
		ax.text(max(x_range) / 2 + 1, 1.05 * max(seq_profile), 'window size: ' + str(window), fontsize=12)

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


def helical_wheel(sequence, colorcoding='rainbow', lineweights=True, filename=None):
	"""A function to project a given peptide sequence onto a helical wheel plot. It can be useful to illustrate the
	properties of alpha-helices, like positioning of charged and hydrophobic residues along the sequence.

	:param sequence: {str} the peptide sequence for which the helical wheel should be drawn.
	:param colorcoding: {str} the color coding to be used, available: *rainbow*, *charge*, *no*
	:param lineweights: {boolean} defines whether connection lines decrease in thickness along the sequence
	:param filename: {str} filename  where to safe the plot. *default = None* --> show the plot
	:return: a helical wheel projection plot of the given sequence (interactively or in **filename**)
	:Example:

	>>> helical_wheel('GLFDIVKKVVGALG')
	>>> helical_wheel('KLLKLLKKLLKLLK', colorcoding='charge')
	>>> helical_wheel('AKLWLKAGRGFGRG', colorcoding='none', lineweights=False)
	>>> helical_wheel('ACDEFGHIKLMNPQRSTVWY')

	.. image:: ../docs/static/wheel1.png
		:scale: 25 %
	.. image:: ../docs/static/wheel2.png
		:scale: 25 %
	.. image:: ../docs/static/wheel3.png
		:scale: 25 %
	.. image:: ../docs/static/wheel4.png
		:scale: 25 %

	.. versionadded:: v2.1.5
	"""
	# color mappings
	aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
	f_rainbow = ['#3e3e28', '#ffcc33', '#b30047', '#b30047', '#ffcc33', '#3e3e28', '#80d4ff', '#ffcc33', '#0047b3',
				 '#ffcc33', '#ffcc33', '#b366ff', '#29a329', '#b366ff', '#0047b3', '#ff66cc', '#ff66cc', '#ffcc33',
				 '#ffcc33', '#ffcc33']
	f_charge = ['#000000', '#000000', '#ff4d94', '#ff4d94', '#000000', '#000000', '#80d4ff', '#000000', '#80d4ff',
				'#000000', '#000000', '#000000', '#000000', '#000000', '#80d4ff', '#000000', '#000000', '#000000',
				'#000000', '#000000']
	f_none = ['#ffffff'] * 20
	t_rainbow = ['w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'w', 'k', 'k', 'k', 'k', 'k', 'w', 'k', 'k', 'k', 'k', 'k']
	t_charge = ['w', 'w', 'k', 'k', 'w', 'w', 'k', 'w', 'k', 'w', 'w', 'w', 'w', 'w', 'k', 'w', 'w', 'w', 'w', 'w']
	t_none = ['k'] * 20
	if lineweights == True:
		lw = np.arange(1, 6, 5. / (len(sequence) - 1))  # line thickness array
		lw = lw[::-1]  # inverse order
	else:
		lw = [2.] * (len(sequence) - 1)

	# check which color coding to use
	if colorcoding == 'rainbow':
		df = dict(zip(aa, f_rainbow))
		dt = dict(zip(aa, t_rainbow))
	elif colorcoding == 'charge':
		df = dict(zip(aa, f_charge))
		dt = dict(zip(aa, t_charge))
	elif colorcoding == 'none':
		df = dict(zip(aa, f_none))
		dt = dict(zip(aa, t_none))
	else:
		print("Unknown color coding, 'rainbow' used instead")
		df = dict(zip(aa, f_rainbow))
		dt = dict(zip(aa, t_rainbow))

	# degree to radian
	deg = np.arange(float(len(sequence))) * -100.
	deg = [d + 90. for d in deg]  # start at 270 degree in unit circle (on top)
	rad = np.radians(deg)

	# create figure
	fig = plt.figure(frameon=False, figsize=(10, 10))
	ax = fig.add_subplot(111)
	old = None

	# iterate over sequence
	for i, r in enumerate(rad):
		if i < 18:
			new = (np.cos(r), np.sin(r))  # new AA coordinates

			# plot the connecting lines
			if old is not None:
				line = lines.Line2D((old[0], new[0]), (old[1], new[1]), transform=ax.transData, color='k',
									linewidth=lw[i - 1])
				line.set_zorder(1)  # 1 = level behind circles
				ax.add_line(line)
		elif i == 18:
			new = (np.cos(r), np.sin(r))
			line = lines.Line2D((old[0], new[0]), (old[1], new[1]), transform=ax.transData, color='k',
								linewidth=lw[i - 1])
			line.set_zorder(1)  # 1 = level behind circles
			ax.add_line(line)
			new = (np.cos(r) * 1.2, np.sin(r) * 1.2)
		else:
			new = (np.cos(r) * 1.2, np.sin(r) * 1.2)

		# plot circles
		circ = patches.Circle(new, radius=0.1, transform=ax.transData, edgecolor='k', facecolor=df[sequence[i]])
		circ.set_zorder(2)  # level in front of lines
		ax.add_patch(circ)

		# check if N- or C-terminus and add subscript, then plot AA letter
		if i == 0:
			ax.text(new[0], new[1], sequence[i] + r'$_N$', va='center', ha='center', transform=ax.transData,
					size=20, color=dt[sequence[i]], fontweight='bold')
		elif i == len(sequence) - 1:
			ax.text(new[0], new[1], sequence[i] + r'$_C$', va='center', ha='center', transform=ax.transData,
					size=20, color=dt[sequence[i]], fontweight='bold')
		else:
			ax.text(new[0], new[1], sequence[i], va='center', ha='center', transform=ax.transData,
					size=20, color=dt[sequence[i]], fontweight='bold')

		old = new  # save as previous coordinates

	# plot shape
	if len(sequence) < 19:
		ax.set_xlim(-1.2, 1.2)
		ax.set_ylim(-1.2, 1.2)
	else:
		ax.set_xlim(-1.4, 1.4)
		ax.set_ylim(-1.4, 1.4)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	cur_axes = plt.gca()
	cur_axes.axes.get_xaxis().set_visible(False)
	cur_axes.axes.get_yaxis().set_visible(False)

	# show or save plot
	if filename:
		plt.savefig(filename, dpi=150)
	else:
		plt.show()


def plot_pde(data, axlabels=None, filename=None):
	"""A function to plot probability density estimations of given data vectors / matrices

	:param data: {np.array} Data array of which underlying probability density function should be estimated and plotted.
	:param axlabels: {list of str} List containing the axis labels for the plot
	:param filename: {str} filename  where to safe the plot. *default = None* --> show the plot
	:Example:

	>>> data = np.random.random([3,100])
	>>> plot_pde(data)

	.. image:: ../docs/static/pde.png
		:scale: 25 %

	.. versionadded:: v2.2.1
	"""

	# colors
	colors = ['#0000ff', '#bf00ff', '#ff0040', '#009900', '#997300']
	if not axlabels:
		axlabels = ['Data', 'Density']

	# transform input to pandas.DataFrame
	data = pd.DataFrame(data)

	fig, ax = plt.subplots()

	# set labels
	ax.set_xlabel(axlabels[0], fontsize=18)
	ax.set_ylabel(axlabels[1], fontsize=18)
	fig.suptitle('Estimated Probability Distribution', fontsize=16, fontweight='bold')

	# only left and bottom axes, no box
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	# plot PDE for every data row
	for i, column in enumerate(data.T):
		# this creates the kernel, given an array it will estimate the probability over that values
		kde = gaussian_kde(data[column])
		# these are the values over which the kernel will be evaluated
		space = np.linspace(0, 1, 1000)
		# plot line
		line = ax.plot(space, kde(space))
		# set line width and color
		plt.setp(line, color=colors[i], linewidth=2.0, alpha=.5)
		# fill area under line
		ax.fill_between(space, 0, kde(space), color=colors[i], alpha=.3)

	# show or save plot
	if filename:
		plt.savefig(filename, dpi=150)
	else:
		plt.show()
