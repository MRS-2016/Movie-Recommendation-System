#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. module: item
   :synopsis: class for creating a single movie item
"""

from . import genre_names

class Item:
    """
    Contains the info for every movie item in the data set.

    One object of :py:class:`Item` contains the information for a single movie.
    
    :param item_data: contains the movie information in the order *movie id*, *movie title*, *release date*, *video release date*, \
                      *IMDb URL*, *unknown*, *Action*, *Adventure*, *Animation*, \
                      *Children's*, *Comedy*, *Crime*, *Documentary*, *Drama*, *Fantasy*, \
                      *Film-Noir*, *Horror*, *Musical*, *Mystery*, *Romance*, *Sci-Fi*, \
                      *Thriller*, *War*, *Western*
    :type item_data: list

    :Example:

    >>> from mrs.datamodel import item
    >>> from mrs import datamodel
    >>> f = open(datamodel.data_location + '/u.item')
    >>> l = (f.readline()).strip().split('|')
    >>> m = item.Item(l)
    >>> print(m)
    id:                 1
    Title:              Toy Story (1995)
    Release Date:       01-Jan-1995
    Video release Date: 
    IMDb URL:           http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)
    Genres:             ["Children's", 'Comedy', 'Animation']

    .. warning:: if you want to run the above example or the file make sure you are out of :py:mod:`mrs` because of relative import statement in :py:mod:`mrs.datamodel.item`
    """
    def __init__(self, item_data):
        self._movie_id           = int(item_data[0])
        self._movie_title        = item_data[1]
        self._movie_release_date = item_data[2]
        self._video_release_date = item_data[3]
        self._IMDb_URL           = item_data[4]
        self._genres             = dict(map(lambda x, y: (x, int(y)), genre_names, item_data[5:]))

    def get_movie_id(self):
        """
        :return: id of the movie
        :rtype:  int
        """
        return self._movie_id

    def get_movie_title(self):
        """
        :return: title of the movie
        :rtype:  str
        """
        return self._movie_title

    def get_movie_release_date(self):
        """
        :return: release data of the movie
        :rtype:  str
        """
        return self._movie_release_date

    def get_video_release_date(self):
        """
        :return: video release date of the item
        :rtype:  str
        """
        return self._video_release_date

    def get_IMDb_URL(self):
        """
        :return: URL of the movie in IMDb site
        :rtype:  str
        """
        return self._IMDb_URL

    def get_genres(self):
        """
        The value of the dict is 1 or 0 depending on if the genre is associated with the movie or not
        
        :return: all the genres information of the movie
        :rtype:  dict
        """
        return self._genres

    def __str__(self):
        """
        pretty print of a movie information
        """
        return "id:                 %s\nTitle:              %s\nRelease Date:       %s\nVideo release Date: %s\nIMDb URL:           %s\nGenres:             %s" % (self._movie_id, self._movie_title, self._movie_release_date,
                                                                                                                                                                   self._video_release_date, self._IMDb_URL, list(filter(lambda x: self._genres[x], self._genres)))
