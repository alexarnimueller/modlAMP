"""
.. module:: plot

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

def plot_feature(y_values, targets=None, y_label='', x_tick_labels=None):
	"""
	Function to generate a box plot of 1 given feature. The different target classes given in **targets** are plottet as separate boxes.

	:param y_values: Array of feature values to be plotted.
	:param targets: List of target class values [string/binary] for the given feature data.
	:param y_label: Axis label.
	:return: A feature box plot.
	"""
	colors = ['dodgerblue','firebrick','gold','lightgreen','navy','black','hotpink'] # available colors

	x = np.array(y_values)
	fig, ax = plt.subplots()

	if targets:
		data = []
		cntr = 0
		for n in list(set(targets)): # finding indices of the different targets in "targets" and plotting
			t = np.array([i for i,j in enumerate(targets) if j == n])
			data.append([x[t]])
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
			labels = 'all data'
		data = x

	# coloring faces of boxes
	medianprops = dict(linestyle='-', linewidth=1, color='black')
	box = ax.boxplot(data,notch=True, patch_artist=True,medianprops=medianprops,labels=labels)
	plt.setp(box['whiskers'], color='black')

	for patch, color in zip(box['boxes'], colors):
		patch.set(facecolor=color,edgecolor='black',alpha=0.8)

	ax.set_xlabel('Classes',fontweight='bold')
	ax.set_ylabel(y_label, fontweight='bold')
	ax.set_title('Feature Box-Plot',fontsize=16,fontweight='bold')
	plt.show()


def plot_2_features(x_values, y_values, targets=None, x_label='', y_label=''):
	"""
	Function to generate a feature scatter plot of 2 given features. The different target classes given in **targets**
	are plottet in different colors.

	:param x_values: Array of values of the feature to be plotted on the x-axis.
	:param y_values: Array of values of the feature to be plotted on the y-axis.
	:param targets: List of target class values [string/binary] for the given feature data.
	:param x_label: X-axis label.
	:param y_label: Y-axis label.
	:return: A 2D feature scatter plot.
	"""

	x = np.array(x_values)
	y = np.array(y_values)
	colors = ['dodgerblue','firebrick','gold','lightgreen','navy','black'] # available colors

	fig, ax = plt.subplots()

	if targets:
		for n in list(set(targets)): # finding indices of the different targets in "targets" and plotting
			t = np.array([i for i,j in enumerate(targets) if j == n])
			xt = x[t] # find all values in x for the given target
			yt = y[t] # find all values in y for the given target
			ax.scatter(xt, yt, c=colors[n], alpha=1., s=25, label='class '+str(n)) # plot scatter for this target group
			ax.legend(loc='best')

	else:
		ax.scatter(x,y,c=colors[0],alpha=1., s=25)

	ax.set_xlabel(x_label, fontweight='bold')
	ax.set_ylabel(y_label, fontweight='bold')
	ax.set_title('2D Feature Plot',fontsize=16,fontweight='bold')
	plt.show()


def plot_3_features(x_values, y_values, z_values, targets=None, x_label='', y_label='', z_label=''):
	"""
	Function to generate a 3D feature scatter plot of 3 given features. The different target classes given in **targets**
	are plottet in different colors.

	:param x_values: Array of values of the feature to be plotted on the x-axis.
	:param y_values: Array of values of the feature to be plotted on the y-axis.
	:param z_values: Array of values of the feature to be plotted on the z-axis.
	:param targets: List of target class values [string/binary] for the given feature data.
	:param x_label: X-axis label.
	:param y_label: Y-axis label.
	:param z_label: Z-axis label.
	:return: A 3D feature scatter plot.
	"""

	x = np.array(x_values)
	y = np.array(y_values)
	z = np.array(z_values)
	colors = ['dodgerblue','firebrick','gold','lightgreen','navy','black'] # available colors

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	if targets:
		for n in list(set(targets)): # finding indices of the different targets in "targets" and plotting
			t = np.array([i for i,j in enumerate(targets) if j == n])
			xt = x[t] # find all values in x for the given target
			yt = y[t] # find all values in y for the given target
			zt = z[t] # find all values in y for the given target
			ax.scatter(xt, yt, zt, c=colors[n], alpha=1., s=25, label='class '+str(n)) # plot 3Dscatter for this target group
			ax.legend(loc='best')

	else:
		ax.scatter(x, y, z, c=colors[0], alpha=1., s=25) # plot 3Dscatter for this target group

	ax.set_xlabel(x_label, fontweight='bold')
	ax.set_ylabel(y_label, fontweight='bold')
	ax.set_zlabel(z_label, fontweight='bold')
	ax.set_title('3D Feature Plot',fontsize=16,fontweight='bold')
	plt.show()