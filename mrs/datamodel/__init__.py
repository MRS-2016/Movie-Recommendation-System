#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. module:: datamodel

This is responsible for loading the data from the data set and creating user and item objects
"""

data_location = '/media/windows7/B.Tech/Project/ml-100k'

user_dataset   = '/u.user'
item_dataset   = '/u.item'
rating_dataset = '/u.data'
info = '/u.info'

genre_names = ["unknown", "Action", "Adventure", "Animation",
               "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
               "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
               "Thriller", "War", "Western"]
