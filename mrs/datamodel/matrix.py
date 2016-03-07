#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from . import data_location, info
from . import nusers, nitems

class Matrix:
    """
    """
    def __init__(self):
        self.rating_matrix = None

    def create_rating_matrix(self, users):
        """
        The first row and first column will be blank and should not be used as the user id and item id starts from 1
        
        :param users: contains user ids as keys and instance of :py:mod:`mrs.datamodel.user.User` as values
        :type users:  dict
        """
        self.rating_matrix = np.full(([nusers + 1, nitems + 1]), fill_value = np.nan)
        
        for user_id, user_profile in users.items():
            for movie, rating in user_profile.get_movie_rating().items():
                self.rating_matrix[user_id, movie] = rating[0]


    def create_rating_matrix_with_missing_as_zero(self, users):
        """
        Same as :py:meth:`create_rating_matrix` except the missing values are zeros instead of nan
        """
        
        self.rating_matrix = np.zeros(([nusers + 1, nitems + 1]))

        for user_id, user_profile in users.items():
            for movie, rating in user_profile.get_movie_rating().items():
                self.rating_matrix[user_id, movie] = rating[0]
