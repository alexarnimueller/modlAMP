# -*- coding: utf-8 -*-
"""
.. module:: modlamp.database

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to connect to several peptide databases. The modlab internal peptide database server
is only available in the CADD intranet at ETH Zurich.
"""

from getpass import getpass

import mysql.connector
import pandas as pd
import requests
from lxml import html
from mysql.connector import Error

__author__ = "modlab"
__docformat__ = "restructuredtext en"


def _read_db_config(host='gsdelta641.ethz.ch', database='peptides', user='modlab', password=None):
    """
    Read database configuration and return a dictionary object.
    This function generally does not need to be used as it is called by the function :func:`query_sequences` directly.

    :param host: The host name of your server hosting the database.
    :param database: Name of the database.
    :param user: Username
    :return: a dictionary of database parameters
    """
    if not password:
        password = getpass()

    db = {'host': host,
          'database': database,
          'user': user,
          'password': password}

    return db


def _connect(conf=None):
    """
    Connect to a given MySQL database in conf.
    This function is called by the function :func:`query_sequences`.

    :param conf: MySQL configuration with host, DB, user and PW. If None, defaults from :py:func:`_read_db_config()`.
    :return: a mysql.connector connection object
    """
    if conf:
        config = _read_db_config(conf)
    else:
        config = _read_db_config()

    try:
        print('Connecting to MySQL database...')
        conn = mysql.connector.connect(**config)
        print('connection established!')
        return conn

    except mysql.connector.Error as err:
        print(err)


def query_database(table, columns=None):
    """
    This function extracts experimental results from the modlab peptide database. All data from the given table and
    column names is extracted and returned.

    :param table: the mysql database table to be queried
    :param columns: a list of the column names {str} to be extracted from the table *default*: '*' (all columns)
    :return: queried data as a numpy array
    :Example:

    >>> data = query_database(table='modlab_experiments', columns=['sequence', 'MCF7_activity', 'Saureus_activity'])
    Password: *********
    Connecting to MySQL database...
    connection established!
    >>> data[:5]
    array([    ['ILGTILGILKGL', None, 1.0],
        ['ILGTILGFLKGL', None, 1.0],
        ['ILGNILGFLKGL', None, 1.0],
        ['ILGQILGILKGL', None, 1.0],
        ['ILGHILGYLKGL', None, 1.0]], dtype=object)

    .. note::
        If 'None' or 'NULL' appears as a value, this means no data was measured for this peptide and not that activity
        is None (inactive).
    """
    if not columns:
        columns = ['*']
    try:
        conn = _connect()
        df = pd.read_sql("SELECT " + ', '.join(columns) + " FROM " + table, con=conn)

        return df

    except Error as e:
        print(e)


def query_apd(id):
    """
    A function to query sequences from the antimicrobial peptide database `APD <http://aps.unmc.edu/AP/>`_.
    If the whole database should be scraped, simpli look up the latest entry ID and take a ``range(1, 'latestID')``
    as function input.
    
    :param id: {list of int} list of APD IDs to be queried from the database
    :return: list of peptide sequences corresponding to entered ids.
    :Example:
    
    >>> query_apd([15, 16, 18, 19, 20])
    ['GLFDIVKKVVGALGSL', 'GLFDIVKKVVGAIGSL', 'GLFDIVKKVVGAFGSL', 'GLFDIAKKVIGVIGSL', 'GLFDIVKKIAGHIAGSI']
    """

    seqs = []

    for i in id:
        page = requests.get('http://aps.unmc.edu/AP/database/query_output.php?ID=%0.5d' % i)
        tree = html.fromstring(page.content)
        seqs.extend(tree.xpath('//font[@color="#ff3300"]/text()'))

    return seqs


def query_camp(id):
    """
    A function to query sequences from the antimicrobial peptide database `CAMP <http://camp.bicnirrh.res.in/>`_.
    If the whole database should be scraped, simpli look up the latest entry ID and take a ``range(1, 'latestID')``
    as function input.

    :param id: {list of int} list of CAMP IDs to be queried from the database
    :return: list of peptide sequences corresponding to entered ids.
    :Example:

    >>> query_camp([2705, 2706])
    ['GLFDIVKKVVGALGSL', 'GLFDIVKKVVGTLAGL']
    """
    
    seqs = []
    
    for i in id:
        page = requests.get('http://camp.bicnirrh.res.in/seqDisp.php?id=CAMPSQ%i' % i)
        tree = html.fromstring(page.content)
        seqs.extend(tree.xpath('//td[@class="fasta"]/text()'))
    
    return seqs
