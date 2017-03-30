# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.database

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to connect to several peptide databases. It also allows to connect to a custom
SQL database for which the configuration is given in a specified config file.
"""

from os.path import exists
import json
from getpass import getpass

import mysql.connector
import pandas as pd
import requests
from lxml import html
from mysql.connector import Error

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


def _read_db_config(configfile):
    """
    Read database configuration and return a dictionary object.
    This function generally does not need to be used as it is called by the function :func:`query_sequences` directly.

    :param configfile: {str} path to the configuration file containing the database information with hostname,
        database name, username and password.
    :return: a dictionary of read database parameters
    """
    if exists(configfile):
        db = json.load(open(configfile, 'r'))
        
        if not db['password']:
            db['password'] = getpass()
    
        return db
    else:
        raise IOError('Path to config file is wrong or file does not exist!\n%s' % configfile)


def _connect(configfile):
    """
    Connect to a given MySQL database in conf. This function is called by the function :func:`query_sequences`.

    :param configfile: path to the MySQL config file containing the hostname, database name, username and password.
        This file is passed to :py:func:`_read_db_config()`.
    :return: a ``mysql.connector`` connection object
    """
    config = _read_db_config(configfile)

    try:
        print('Connecting to MySQL database...')
        conn = mysql.connector.connect(**config)
        print('connection established!')
        return conn

    except mysql.connector.Error as err:
        print(err)


def query_database(table, columns=None, configfile='./modlamp/data/db_config.json'):
    """
    This function extracts experimental results from the modlab peptide database. All data from the given table and
    column names is extracted and returned.

    :param table: the mysql database table to be queried
    :param columns: a list of the column names {str} to be extracted from the table *default*: ``*`` (all columns)
    :param configfile: location of the database configuration file containing the hostname etc. for the database to
        be queried.
    :return: {numpy.array} queried data
    :Example:

    >>> data = query_database(table='modlab_experiments', columns=['sequence', 'MCF7_activity', 'Saureus_activity'])
    Password: *********
    Connecting to MySQL database...
    connection established!
    >>> data[:5]
    array([ ['ILGTILGILKGL', None, 1.0],
            ['ILGTILGFLKGL', None, 1.0],
            ['ILGNILGFLKGL', None, 1.0],
            ['ILGQILGILKGL', None, 1.0],
            ['ILGHILGYLKGL', None, 1.0]], dtype=object)

    .. note::
        If ``None`` or ``NULL`` appears as a value, this means no data was measured for this peptide and not that
        activity is none (inactive).
    """
    if not columns:
        columns = ['*']
    try:
        conn = _connect(configfile)
        df = pd.read_sql("SELECT " + ', '.join(columns) + " FROM " + table, con=conn)

        return df

    except Error as e:
        print(e)


def query_apd(ids):
    """
    A function to query sequences from the antimicrobial peptide database `APD <http://aps.unmc.edu/AP/>`_.
    If the whole database should be scraped, simply look up the latest entry ID and take a ``range(1, 'latestID')``
    as function input.
    
    :param ids: {list of int} list of APD IDs to be queried from the database
    :return: list of peptide sequences corresponding to entered ids.
    :Example:
    
    >>> query_apd([15, 16, 18, 19, 20])
    ['GLFDIVKKVVGALGSL', 'GLFDIVKKVVGAIGSL', 'GLFDIVKKVVGAFGSL', 'GLFDIAKKVIGVIGSL', 'GLFDIVKKIAGHIAGSI']
    """

    seqs = []

    for i in ids:
        page = requests.get('http://aps.unmc.edu/AP/database/query_output.php?ID=%0.5d' % i)
        tree = html.fromstring(page.content)
        seqs.extend(tree.xpath('//font[@color="#ff3300"]/text()'))

    return seqs


def query_camp(ids):
    """
    A function to query sequences from the antimicrobial peptide database `CAMP <http://camp.bicnirrh.res.in/>`_.
    If the whole database should be scraped, simply look up the latest entry ID and take a ``range(1, 'latestID')``
    as function input.

    :param ids: {list of int} list of CAMP IDs to be queried from the database
    :return: list of peptide sequences corresponding to entered ids.
    :Example:

    >>> query_camp([2705, 2706])
    ['GLFDIVKKVVGALGSL', 'GLFDIVKKVVGTLAGL']
    """
    
    seqs = []
    
    for i in ids:
        page = requests.get('http://camp.bicnirrh.res.in/seqDisp.php?id=CAMPSQ%i' % i)
        tree = html.fromstring(page.content)
        seqs.extend(tree.xpath('//td[@class="fasta"]/text()'))
    
    return seqs
