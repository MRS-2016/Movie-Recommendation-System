#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""this module will act as an intermediate through which other classes will get the data from"""

from . import data_location
from . import user_dataset, item_dataset, rating_dataset

from .user import User
from .item import Item
from .matrix import Matrix

class Data:
    """
    Loads the data from the dataset and puts the data in to variables for later processing.
    Any algorithm wants to access the data can call the function from this class to access a particular type of data.
    :py:attr:`users` is a dict which contains the user ids as keys and instance of :py:class:`mrs.datamodel.user.User` as values
    :py:attr:`items` is a dict which contains the item ids as keys and instance of :py:class:`mrs.datamodel.item.Item` as values
    
    .. note:: This class should act as an intermediate to access the data from the dataset. If any other type of data is needed then write a function in this class to convert the data into that type and return it
    
    .. todo:: if possible write the attributes in unordered list format
    """
    def __init__(self):
        self._users   = {}
        self._items   = {}

    def load_data(self):
        """
        loads the dataset from *u.data* to :py:attr:`_users` and *u.item* to :py:attr:`_items` and adds the rating given by a user to its instance
        """
        try:
            for data in open(data_location + user_dataset):
                l = data.strip().split('|')
                self._users[int(l[0])] = User(l)
        except FileNotFoundError:
            print("The file couldn't be located")

        try:
            for data in open(data_location + item_dataset, encoding = 'CP1252'):
                l = data.strip().split('|')
                self._items[int(l[0])] = Item(l)
        except FileNotFoundError:
            print("The file couldn't be located")

        try:
            for data in open(data_location + rating_dataset):
                l = [int(val) for val in data.strip().split()]
                self._users[l[0]].add_movie(l[1:])
        except FileNotFoundError:
            print("The file couldn't be located")

    def get_users(self):
        """
        :return: :py:attr:`_users`
        :rtype:  dict
        """
        return self._users

    def get_items(self):
        """
        :return: :py:attr:`_items`
        :rtype:  dict
        """
        return self._items

    def get_rating_matrix(self):
        """
        :return: :py:attr:`mrs.datamodel.matrix.Matrix.rating_matrix`
        :rtype:  ndarray
        :Example:
        >>> from mrs.datamodel import loaddata
        >>> d = loaddata.Data()
        >>> r = d.get_rating_matrix()
        >>> r[196, 242]
        3.0
        >>> r[186, 302]
        3.0
        >>> r[22, 377]
        1.0
        >>> r[244, 51]
        2.0
        """
        m = Matrix()
        m.create_rating_matrix(self._users)
        return m.rating_matrix
