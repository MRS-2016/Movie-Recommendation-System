#!/usr/bin/env python
# -*- coding: utf-8 -*-

class User:
    """
    One object of :py:class:`User` contains the information for a single user
    
    :param user_data: contains user information in the order *user_id*, *age*, *gender*, *occupation*, *zip code*
    :type user_data: list

    :Example:

    >>> from mrs.datamodel import user
    >>> from mrs import datamodel
    >>> f = open(datamodel.data_location + '/u.user')
    >>> l = (f.readline()).strip().split('|')
    >>> u = user.User(l)
    >>> print(u)
    id:         1
    age:        24
    gender:     M
    occupation: technician
    zip code:   85711
    """
    def __init__(self, user_data):
        self._user_id     = int(user_data[0])
        self._age         = int(user_data[1])
        self._gender      = user_data[2]
        self._occupation  = user_data[3]
        self._zip_code    = user_data[4]
        self._movie_rating = {}

    def get_id(self):
        """
        :return: the id of the user
        :rtype:  int
        """
        return self._user_id

    def get_age(self):
        """
        :return: the age of the user
        :rtype:  int
        """
        return self._age

    def get_gender(self):
        """
        :return: the gender of the user
        :rtype:  str
        """
        return self._gender

    def get_occupation(self):
        """
        :return: the occupation of the user
        :rtype:  str
        """
        return self._occupation

    def get_zip_code(self):
        """
        :return: the zip code of the location where the user lives
        :rtype:  str
        """
        return self._zip_code

    def get_movie_rating(self):
        """
        :return: the ratings for movies by this user
        :rtype:  dict
        """
        return self._movie_rating

    def __str__(self):
        """
        pretty print of user information
        """
        return "id:         %d\nage:        %d\ngender:     %s\noccupation: %s\nzip code:   %s" % (self._user_id, self._age, self._gender, self._occupation, self._zip_code)

    def add_movie(self, data):
        """
        Adds to the user's movie collection by adding the movie id and rating to :py:attr:`movie_rating`
        
        :param data: contains the movie id, rating given to the movie by this user along with timestamp
        :type data: list
        """
        self._movie_rating[data[0]] = data[1:]

    def mean_rating(self):
        """
        returns the mean rating of this user
        """
        return sum(map(lambda x: x[0], self._movie_rating.values())) / len(self._movie_rating)
