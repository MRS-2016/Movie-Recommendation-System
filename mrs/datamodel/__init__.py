#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. module:: datamodel

This is responsible for loading the data from the data set and creating user and item objects
"""

# path to the data set
data_location = '/media/windows7/B.Tech/Project/ml-100k'

user_dataset   = '/u.user'
item_dataset   = '/u.item'
rating_dataset = '/u.data'
info = '/u.info'

# load the number of users, items and ratings
l = open(data_location + info).readlines()
nusers   = int(l[0].split()[0])
nitems   = int(l[1].split()[0])
nratings = int(l[2].split()[0])

genre_names = ["unknown", "Action", "Adventure", "Animation",
               "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
               "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
               "Thriller", "War", "Western"]
