# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.plot

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to plot different feature plots. The following functions are available:

============================        ==============================================================================
Function                            Characteristics
============================        ==============================================================================
:py:func:`plot_feature`             Generate a box plot for visualizing the distribution of a given feature.
:py:func:`plot_2_features`          Generate a 2D scatter plot of 2 given features.
:py:func:`plot_3_features`          Generate a 3D scatter plot of 3 given features.
:py:func:`plot_profile`             Generates a profile plot of a sequence to visualize potential linear gradients.
:py:func:`helical_wheel`            Generates a helical wheel projection plot of a given sequence.
:py:func:`plot_pde`                 Generates a probability density estimation plot of given data arrays.
:py:func:`plot_violin`              Generates a violin plot for given classes and corresponding distributions.
:py:func:`plot_aa_distr`            Generates an amino acid frequency plot for all 20 natural amino acids.
============================        ==============================================================================

"""
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats.kde import gaussian_kde

from modlamp.core import count_aas, load_scale
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


def plot_feature(y_values, targets=None, y_label='feature values', x_tick_labels=None, filename=None, colors=None):
    """
    Function to generate a box plot of 1 given feature. The different target classes given in **targets** are plottet
    as separate boxes.

    :param y_values: Array of feature values to be plotted.
    :param targets: List of target class values [string/binary] for the given feature data.
    :param y_label: Axis label.
    :param x_tick_labels: list of labels to be assigned to the ticks on the x-axis. Must match the number of targets.
    :param filename: filename where to safe the plot. *default = None*
    :param colors: {list} colors to take for plotting (strings in HEX formats).
    :return: A feature box plot.
    :Example:

    >>> plot_feature(desc.descriptor,y_label='uH Eisenberg')  # desc: PeptideDescriptor instance

    .. image:: ../docs/static/uH_Eisenberg.png
        :height: 300px
    
    The same procedure also works for comparing two data sets:
    
    >>> plot_feature((p.descriptor, apd.descriptor), y_label='uH Eisenberg', x_tick_labels=['Library', 'APD3'])
    
    .. image:: ../docs/static/uH_APD3.png
        :height: 300px
    """
    if not colors:
        colors = ['#69D2E7', '#FA6900', '#E0E4CC', '#542437', '#53777A', 'black', '#C02942', '#031634']

    if type(y_values) == list:
        y_values = np.array(y_values)

    if len(targets) >= 1:
        data = []
        cntr = 0
        for n in set(targets):  # finding indices of the different targets in "targets" and plotting
            data.append(y_values[np.where(targets == n)])
            cntr += 1

        if x_tick_labels:
            labels = x_tick_labels
        else:
            labels = [str(i) for i in range(cntr)]

        colors = colors[:cntr]
    
    else:
        if x_tick_labels:
            labels = x_tick_labels
        else:
            labels = ['all data']
        data = y_values

    fig, ax = plt.subplots()
    # coloring faces of boxes
    median_props = dict(linestyle='-', linewidth='1', color='black')
    box = ax.boxplot(data, notch=True, patch_artist=True, medianprops=median_props, labels=labels)
    plt.setp(box['whiskers'], color='black')
    
    for patch, color in zip(box['boxes'], colors):
        patch.set(facecolor=color, edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Classes', fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title('Feature Box-Plot', fontsize=16, fontweight='bold')
    # only left and bottom axes, no box
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def plot_2_features(x_values, y_values, targets=None, x_label='', y_label='', filename=None, colors=None):
    """
    Function to generate a feature scatter plot of 2 given features. The different target classes given in **targets**
    are plottet in different colors.

    :param x_values: Array of values of the feature to be plotted on the x-axis.
    :param y_values: Array of values of the feature to be plotted on the y-axis.
    :param targets: List of target class values [string/binary] for the given feature data.
    :param x_label: X-axis label.
    :param y_label: Y-axis label.
    :param filename: filename where to safe the plot. *default = None*
    :param colors: {list} colors to take for plotting (strings in HEX formats).
    :return: A 2D feature scatter plot.
    :Example:

    >>> plot_2_features(a.descriptor,b.descriptor,x_label='uH',y_label='pI',targets=targs)

    .. image:: ../docs/static/2D_scatter.png
        :height: 300px
    """
    if not colors:
        colors = ['#69D2E7', '#FA6900', '#B5B8AB', '#542437', '#53777A', 'black', '#C02942', '#031634']
    
    fig, ax = plt.subplots()
    
    if len(targets) >= 1:
        for n in list(set(targets)):  # finding indices of the different targets in "targets" and plotting
            t = np.array([i for i, j in enumerate(targets) if j == n])
            xt = x_values[t]  # find all values in x for the given target
            yt = y_values[t]  # find all values in y for the given target
            ax.scatter(xt, yt, c=colors[n], alpha=1., s=25,
                       label='class ' + str(n))  # plot scatter for this target group
            ax.legend(loc='best')
    
    else:
        ax.scatter(x_values, y_values, c=colors[0], alpha=1., s=25)
    
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title('2D Feature Plot', fontsize=16, fontweight='bold')
    # only left and bottom axes, no box
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def plot_3_features(x_values, y_values, z_values, targets=None, x_label='', y_label='', z_label='', filename=None,
                    colors=None):
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
    :param colors: {list} colors to take for plotting (strings in HEX formats).
    :return: A 3D feature scatter plot.
    :Example:

    >>> plot_3_features(a.descriptor,b.descriptor,c.descriptor,x_label='uH',y_label='pI',z_label='length')

    .. image:: ../docs/static/3D_scatter.png
        :height: 300px
    """
    
    if not colors:
        colors = ['#69D2E7', '#FA6900', '#E0E4CC', '#542437', '#53777A', 'black', '#C02942', '#031634']
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if len(targets) >= 1:
        for n in list(set(targets)):  # finding indices of the different targets in "targets" and plotting
            t = np.array([i for i, j in enumerate(targets) if j == n])
            xt = x_values[t]  # find all values in x for the given target
            yt = y_values[t]  # find all values in y for the given target
            zt = z_values[t]  # find all values in y for the given target
            ax.scatter(xt, yt, zt, c=colors[n], alpha=1., s=25,
                       label='class ' + str(n))  # plot 3Dscatter for this target
            ax.legend(loc='best')
    
    else:  # plot 3D scatter for this target group
        ax.scatter(x_values, y_values, z_values, c=colors[0], alpha=1., s=25)
    
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_zlabel(z_label, fontweight='bold')
    ax.set_title('3D Feature Plot', fontsize=16, fontweight='bold')
    # only left and bottom axes, no box
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('bottom')
    
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def plot_profile(sequence, window=5, scalename='Eisenberg', filename=None, color='red', seq=False, ylim=None):
    """ Function to generate sequence profile plots of a given amino acid scale or a moment thereof.

    .. note::
        :func:`plot_profile` can only plot one-dimensional amino acid scales given in
        :class:`modlamp.descriptors.PeptideDescriptor`.

    :param sequence: {str} Peptide sequence for which the profile should be plotted.
    :param window: {int, uneven} Window size for which the average value is plotted for the center amino acid.
    :param scalename: {str} Amino acid scale to be used to describe the sequence.
    :param filename: {str} Filename  where to safe the plot. *default = None* --> show the plot
    :param color: {str} Color of the plot line.
    :param seq: {bool} Whether the amino acid sequence should be plotted as the title.
    :param ylim: {tuple of float} Y-Axis limits. Provide as tuple, e.g. (0.5, -0.2)
    :return: a profile plot of the input sequence interactively or with the specified *filename*
    :Example:

    >>> plot_profile('GLFDIVKKVVGALGSL', scalename='eisenberg')

    .. image:: ../docs/static/profileplot.png
        :height: 300px

    .. versionadded:: v2.1.5
    """
    # check if given scale is defined in PeptideDescriptor
    d = PeptideDescriptor(sequence, scalename)
    if len(d.scale['A']) > 1:
        raise KeyError("\nSorry\nThis function can only calculate profiles for 1D scales. '%s' has more than one "
                       "dimension" % scalename)
    seq_data = list()
    seq_profile = list()
    for a in sequence:
        seq_data.append(d.scale[a])  # describe sequence by given scale
    i = 0  # AA index
    while (i + window) < len(sequence):
        seq_profile.append(np.mean(seq_data[i:(i + window + 1)]))  # append average value for given window
        i += 1

    # plot
    fig, ax = plt.subplots()
    x_range = range(int(window) / 2 + 1, len(sequence) - int(window) / 2)
    line = ax.plot(x_range, seq_profile)
    plt.setp(line, color=color, linewidth=2.0)

    # axis labes and title
    ax.set_xlabel('sequence position', fontweight='bold')
    ax.set_ylabel(scalename + ' value', fontweight='bold')
    ax.text(max(x_range) / 2 + 1, 1.05 * max(seq_profile), 'window size: ' + str(window),
            fontsize=16, fontweight='bold')
    if seq:
        ax.set_title(sequence, fontsize=16, fontweight='bold', y=1.02)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(1.2 * max(seq_profile), 1.2 * min(seq_profile))

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


def helical_wheel(sequence, colorcoding='rainbow', lineweights=True, filename=None, seq=False, moment=False):
    """A function to project a given peptide sequence onto a helical wheel plot. It can be useful to illustrate the
    properties of alpha-helices, like positioning of charged and hydrophobic residues along the sequence.

    :param sequence: {str} the peptide sequence for which the helical wheel should be drawn
    :param colorcoding: {str} the color coding to be used, available: *rainbow*, *charge*, *polar*, *simple*,
        *amphipathic*, *none*
    :param lineweights: {boolean} defines whether connection lines decrease in thickness along the sequence
    :param filename: {str} filename  where to safe the plot. *default = None* --> show the plot
    :param seq: {bool} whether the amino acid sequence should be plotted as a title
    :param moment: {bool} whether the Eisenberg hydrophobic moment should be calculated and plotted
    :return: a helical wheel projection plot of the given sequence (interactively or in **filename**)
    :Example:

    >>> helical_wheel('GLFDIVKKVVGALG')
    >>> helical_wheel('KLLKLLKKLLKLLK', colorcoding='charge')
    >>> helical_wheel('AKLWLKAGRGFGRG', colorcoding='none', lineweights=False)
    >>> helical_wheel('ACDEFGHIKLMNPQRSTVWY')

    .. image:: ../docs/static/wheel1.png
        :height: 300px
    .. image:: ../docs/static/wheel2.png
        :height: 300px
    .. image:: ../docs/static/wheel3.png
        :height: 300px
    .. image:: ../docs/static/wheel4.png
        :height: 300px

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
    f_polar = ['#000000', '#000000', '#80d4ff', '#80d4ff', '#000000', '#000000', '#80d4ff', '#000000', '#80d4ff',
               '#000000', '#000000', '#80d4ff', '#000000', '#80d4ff', '#80d4ff', '#80d4ff', '#80d4ff', '#000000',
               '#000000', '#000000']
    f_simple = ['#ffcc33', '#ffcc33', '#0047b3', '#0047b3', '#ffcc33', '#7f7f7f', '#0047b3', '#ffcc33', '#0047b3',
                '#ffcc33', '#ffcc33', '#0047b3', '#ffcc33', '#0047b3', '#0047b3', '#0047b3', '#0047b3', '#ffcc33',
                '#ffcc33', '#ffcc33']
    f_none = ['#ffffff'] * 20
    f_amphi = ['#ffcc33', '#29a329', '#b30047', '#b30047', '#f79318', '#80d4ff', '#0047b3', '#ffcc33', '#0047b3',
               '#ffcc33', '#ffcc33', '#80d4ff', '#29a329', '#80d4ff', '#0047b3', '#80d4ff', '#80d4ff', '#ffcc33',
               '#f79318', '#f79318']
    t_rainbow = ['w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'w', 'k', 'k', 'k', 'k', 'k', 'w', 'k', 'k', 'k', 'k', 'k']
    t_charge = ['w', 'w', 'k', 'k', 'w', 'w', 'k', 'w', 'k', 'w', 'w', 'w', 'w', 'w', 'k', 'w', 'w', 'w', 'w', 'w']
    t_polar = ['w', 'w', 'k', 'k', 'w', 'w', 'k', 'w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'k', 'k', 'w', 'w', 'w']
    t_simple = ['k', 'k', 'w', 'w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'k', 'k', 'w', 'w', 'w', 'w', 'k', 'k', 'k']
    t_none = ['k'] * 20
    t_amphi = ['k', 'k', 'w', 'w', 'w', 'k', 'w', 'k', 'w', 'k', 'k', 'k', 'w', 'k', 'w', 'k', 'k', 'k', 'w', 'w']
    d_eisberg = load_scale('eisenberg')[1]  # eisenberg hydrophobicity values for HM
    
    if lineweights:
        lw = np.arange(0.1, 5.5, 5. / (len(sequence) - 1))  # line thickness array
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
    elif colorcoding == 'polar':
        df = dict(zip(aa, f_polar))
        dt = dict(zip(aa, t_polar))
    elif colorcoding == 'simple':
        df = dict(zip(aa, f_simple))
        dt = dict(zip(aa, t_simple))
    elif colorcoding == 'none':
        df = dict(zip(aa, f_none))
        dt = dict(zip(aa, t_none))
    elif colorcoding == 'amphipathic':
        df = dict(zip(aa, f_amphi))
        dt = dict(zip(aa, t_amphi))
    else:
        print("Unknown color coding, 'rainbow' used instead")
        df = dict(zip(aa, f_rainbow))
        dt = dict(zip(aa, t_rainbow))
    
    # degree to radian
    deg = np.arange(float(len(sequence))) * -100.
    deg = [d + 90. for d in deg]  # start at 270 degree in unit circle (on top)
    rad = np.radians(deg)
    
    # dict for coordinates and eisenberg values
    d_hydro = dict(zip(rad, [0.] * len(rad)))
    
    # create figure
    fig = plt.figure(frameon=False, figsize=(10, 10))
    ax = fig.add_subplot(111)
    old = None
    hm = list()
    
    # iterate over sequence
    for i, r in enumerate(rad):
        new = (np.cos(r), np.sin(r))  # new AA coordinates
        if i < 18:
            # plot the connecting lines
            if old is not None:
                line = lines.Line2D((old[0], new[0]), (old[1], new[1]), transform=ax.transData, color='k',
                                    linewidth=lw[i - 1])
                line.set_zorder(1)  # 1 = level behind circles
                ax.add_line(line)
        elif 17 < i < 36:
            line = lines.Line2D((old[0], new[0]), (old[1], new[1]), transform=ax.transData, color='k',
                                linewidth=lw[i - 1])
            line.set_zorder(1)  # 1 = level behind circles
            ax.add_line(line)
            new = (np.cos(r) * 1.2, np.sin(r) * 1.2)
        elif i == 36:
            line = lines.Line2D((old[0], new[0]), (old[1], new[1]), transform=ax.transData, color='k',
                                linewidth=lw[i - 1])
            line.set_zorder(1)  # 1 = level behind circles
            ax.add_line(line)
            new = (np.cos(r) * 1.4, np.sin(r) * 1.4)
        else:
            new = (np.cos(r) * 1.4, np.sin(r) * 1.4)
        
        # plot circles
        circ = patches.Circle(new, radius=0.1, transform=ax.transData, edgecolor='k', facecolor=df[sequence[i]])
        circ.set_zorder(2)  # level in front of lines
        ax.add_patch(circ)
        
        # check if N- or C-terminus and add subscript, then plot AA letter
        if i == 0:
            ax.text(new[0], new[1], sequence[i] + b'$_N$', va='center', ha='center', transform=ax.transData,
                    size=32, color=dt[sequence[i]], fontweight='bold')
        elif i == len(sequence) - 1:
            ax.text(new[0], new[1], sequence[i] + b'$_C$', va='center', ha='center', transform=ax.transData,
                    size=32, color=dt[sequence[i]], fontweight='bold')
        else:
            ax.text(new[0], new[1], sequence[i], va='center', ha='center', transform=ax.transData,
                    size=36, color=dt[sequence[i]], fontweight='bold')
        
        eb = d_eisberg[sequence[i]][0]  # eisenberg value for this AA
        hm.append([eb * new[0], eb * new[1]])  # save eisenberg hydrophobicity vector value to later calculate HM
        
        old = (np.cos(r), np.sin(r))  # save as previous coordinates
    
    # draw hydrophobic moment arrow if moment option
    if moment:
        v_hm = np.sum(np.array(hm), 0)
        x = .0333 * v_hm[0]
        y = .0333 * v_hm[1]
        ax.arrow(0., 0., x, y, head_width=0.04, head_length=0.03, transform=ax.transData,
                 color='k', linewidth=6.)
        desc = PeptideDescriptor(sequence)  # calculate hydrophobic moment
        desc.calculate_moment()
        if abs(x) < 0.2 and y > 0.:  # right positioning of HM text so arrow does not cover it
            z = -0.2
        else:
            z = 0.2
        plt.text(0., z, str(round(desc.descriptor[0][0], 3)), fontdict={'fontsize': 20, 'fontweight': 'bold',
                                                                        'ha': 'center'})
    
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
    plt.tight_layout()
    
    if seq:
        plt.title(sequence, fontweight='bold', fontsize=20)
    
    # show or save plot
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def plot_pde(data, title=None, axlabels=None, filename=None, legendloc=2, x_min=0, x_max=1, colors=None, alpha=0.2):
    """A function to plot probability density estimations of given data vectors / matrices (row wise)

    :param data: {list / array} data of which underlying probability density function should be estimated and plotted.
    :param title: {str} plot title
    :param axlabels: {list of str} list containing the axis labels for the plot
    :param filename: {str} filename  where to safe the plot. *default = None* --> show the plot
    :param legendloc: {int} location of the figures legend. 1 = top right, 2 = top left ...
    :param x_min: {number} x-axis minimum
    :param x_max: {number} x-axis maximum
    :param colors: {list} list of colors (readable by matplotlib, e.g. hex) to be used to plot different data classes
    :param alpha: {float} color alpha for filling pde curve
    :Example:

    >>> data = np.random.random((3,100))
    >>> plot_pde(data)

    .. image:: ../docs/static/pde.png
        :height: 300px

    .. versionadded:: v2.2.1
    """
    if not axlabels:
        axlabels = ['Data', 'Estimated Density']
    if not title:
        title = ""
    
    # transform input to numpy array and reshape if it only contains one data row
    data = np.array(data)
    
    if len(data.shape) == 1:
        data = data.reshape((1, -1))
    shp = data.shape
    
    # colors
    if not colors:
        colors = ['#0B486B', '#3B8686', '#79BD9A', '#A8DBA8', '#CFF09E', '#0000ff', '#bf00ff', '#ff0040', '#009900']
    elif len(colors) != len(data) and shp != 1:  # if not enough colors for all data subtypes
        colors *= len(data)
    
    # prepare figure
    fig, ax = plt.subplots()
    
    # set axis labels and limits
    if axlabels is None:
        axlabels = ['', '']
    ax.set_xlabel(axlabels[0], fontsize=18)
    ax.set_ylabel(axlabels[1], fontsize=18)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # only left and bottom axes, no box
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    # plot PDE for every data row
    # if one row only
    if shp[0] == 1:
        kde = gaussian_kde(
                data)  # this creates the kernel, given an array it will estimate the probability over that values
        space = np.linspace(x_min, x_max, 1000)  # these are the values over which the kernel will be evaluated
        line = ax.plot(space, kde(space), label='Data')  # plot line
        plt.setp(line, color=colors[0], linewidth=2.0, alpha=.9)  # set line width and color
        ax.fill_between(space, 0, kde(space), color=colors[0], alpha=alpha)  # fill area under line
    
    # if multiple rows
    else:
        for i, row in enumerate(data):
            kde = gaussian_kde(
                    row)  # this creates the kernel, given an array it will estimate the probability over that values
            space = np.linspace(x_min, x_max, 1000)  # these are the values over which the kernel will be evaluated
            line = ax.plot(space, kde(space), label='Run ' + str(i))  # plot line
            plt.setp(line, color=colors[i], linewidth=2.0, alpha=.9)  # set line width and color
            ax.fill_between(space, 0, kde(space), color=colors[i], alpha=alpha)  # fill area under line
    
    # show or save plot
    ax.legend(loc=legendloc)
    ax.set_xlim((x_min, x_max))
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def plot_violin(x, colors=None, bp=False, filename=None, title=None, axlabels=None, y_min=0, y_max=1):
    """    create violin plots out of given data array
    (adapted from `Flavio Coelho <https://pyinsci.blogspot.ch/2009/09/violin-plot-with-matplotlib.html>`_.)

    :param x: {numpy.array} data to be plotted
    :param colors: {str or list} face color of the violin plots, can also be list of colors with same dimension as **x**
    :param bp: {bool} print a box blot inside violin
    :param filename: {str} location / filename where to save the plot to. *default = None* --> show the plot
    :param title: {str} Title of the plot.
    :param axlabels: {list of str} list containing the axis labels for the plot
    :param y_min: {number} y-axis minimum.
    :param y_max: {number} y-axis maximum.
    :Example:

    >>> data = np.random.normal(size=[5, 100])
    >>> plot_violin(data, colors=['#0B486B', '#0B486B', '#0B486B', '#CFF09E', '#CFF09E'], bp=True, y_min=-3, y_max=3)

    .. image:: ../docs/static/violins.png
        :height: 300px

    .. versionadded:: v2.2.2
    """
    
    # transform input to list of arrays (better handled by plotting functions)
    x = np.array(x)
    
    # check color input and transform to list of right length
    if not colors:
        colors = ['#0B486B', '#3B8686', '#79BD9A', '#A8DBA8', '#CFF09E', '#0000ff', '#bf00ff', '#ff0040', '#009900']
    
    if isinstance(colors, basestring):
        colors = [colors] * len(x)
    
    # scaling for available space
    dist = len(x) - 1
    w = min(0.15 * max(dist, 1.0), 0.5)
    
    fig, ax = plt.subplots()
    if len(np.array(x).shape) == 1:  # if only one dimensional data
        k = gaussian_kde(x)  # kernel density estimation
        mi = k.dataset.min()  # lower bound of violin
        ma = k.dataset.max()  # upper bound of violin
        rng = np.arange(mi, ma, (ma - mi) / 100.)  # range over which the PDE is performed
        v = k.evaluate(rng)  # violin profile (density curve)
        v = v / v.max() * 0.3  # scaling the violin to the available space
        ax.fill_betweenx(rng, 1, v + 1, facecolor=colors[0], alpha=0.6)
        ax.fill_betweenx(rng, 1, -v + 1, facecolor=colors[0], alpha=0.6)
        
        if bp:  # print box plot if option is given
            medprops = dict(linestyle='-', linewidth='1', color='black')
            box = ax.boxplot(x, notch=1, positions=[1.], vert=1, patch_artist=True, medianprops=medprops)
            plt.setp(box['whiskers'], color='black')
            box['boxes'][0].set(facecolor=colors[0], edgecolor='black', alpha=0.7)
    
    else:  # one violin for every data element if multidimensional
        for p, d in enumerate(x):
            loc = p + 1
            k = gaussian_kde(d)  # kernel density estimation
            mi = k.dataset.min()  # lower bound of violin
            ma = k.dataset.max()  # upper bound of violin
            rng = np.arange(mi, ma, (ma - mi) / 100.)  # range over which the PDE is performed
            v = k.evaluate(rng)  # violin profile (density curve)
            v = v / v.max() * w  # scaling the violin to the available space
            ax.fill_betweenx(rng, loc, v + loc, facecolor=colors[p], alpha=0.6)
            ax.fill_betweenx(rng, loc, -v + loc, facecolor=colors[p], alpha=0.6)
        
        if bp:  # print box plots if option is given
            box = ax.boxplot(x.T, notch=1, vert=1, patch_artist=True)
            plt.setp(box['whiskers'], color='black')
            plt.setp(box['medians'], linestyle='-', linewidth=1.5, color='black')
            for p, patch in enumerate(box['boxes']):
                patch.set(facecolor=colors[p], edgecolor='black', alpha=0.7)
    
    # only left and bottom axes, no box
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='x', which='both', top='off')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim((y_min, y_max))
    if axlabels is None:
        axlabels = ['', '']
    ax.set_xlabel(axlabels[0], fontsize=18)
    ax.set_ylabel(axlabels[1], fontsize=18)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title('Violin Plots', fontsize=16, fontweight='bold')
    
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def plot_aa_distr(sequences, color='#83AF9B', filename=None):
    """Method to plot the amino acid distribution of a given list of sequences

    :param sequences: {list} list of sequences to calculate the amino acid distribution fore
    :param color: {str} color to be used (matplotlib style / hex)
    :param filename: {str} location / filename where to save the plot to. *default = None* --> show the plot
    :Example:

    >>> plot_aa_distr(['KLLKLLKKLLKLLK', 'WWRRWWRAARWWRRWWRR', 'ACDEFGHKLCMNPQRSTVWY', 'GGGGGIIKLWGGGGGGGGGGGGG'])

    .. image:: ../docs/static/AA_dist.png
        :height: 300px

    .. versionadded:: v2.2.5
    """
    concatseq = ''.join(sequences)
    aa = count_aas(concatseq, scale='relative')
    
    fig, ax = plt.subplots()
    
    for a in range(20):
        plt.bar(a, aa.values()[a], 0.9, color=color)
    
    plt.xlim([-0.75, 19.75])
    plt.ylim([0, max(aa.values()) + 0.05])
    plt.xticks(range(20), aa.keys(), fontweight='bold')
    plt.ylabel('Amino Acid Frequency', fontweight='bold')
    plt.title('Amino Acid Distribution', fontsize=16, fontweight='bold')
    
    # only left and bottom axes, no box
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    if filename:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
