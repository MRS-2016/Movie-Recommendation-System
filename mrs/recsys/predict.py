#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this module is responsible for loading the data by calling the necessary information from **loaddata** module from datamodel directory and then calling necessary algorithms to predict the score
# note that this will act as intermediary file for giving result by interacting with all other files

# third party packages
import numpy as np

# local files
from ..datamodel import loaddata, nitems, nusers
from . import cf, ann, convert

class Predict:
    """
    It is responsible for predicting scores for an user to an item by using different techniques.

    Not matter what technique you use you will load the data first so it makes sense to load the data in __init__ method. Different techniques use different algorithms to calculate the prediction, so each method have its own separate implementation.
    """
    def __init__(self):
        self.data = loaddata.Data()
        self.data.load_data()

    @staticmethod
    def scale(l):
        l = l[0]
        no, val = 1, l[0]
        for i, x in enumerate(l):
            if x > val:
                no = i + 1
                val = x
        return no

    @staticmethod
    def f(rating):
        """
        converts the rating to a format for a five output neuron
        
        :param rating: its a rating value given to a movie item
        :type rating:  int
        """
        l = [0, 0, 0, 0, 0]
        l[rating - 1] = 1
        return l

class PredictNeuralNetwork(Predict):
    def __init__(self):
        Predict.__init__(self)
        self.rating_matrix      = self.data.get_rating_matrix_with_nan()
        self.correlation_matrix = cf.Correlation().pearson(self.rating_matrix)

    def create_training_examples_with_item(self, ratings):
        """
        :param ratings: list of tuples with item_id at 0th index and rating at 1th index
        :type ratings:  list
        :return: list of tuples of training examples and each training example contains item feature and rating given to that item
        :rtype:  list
        """
        feature = []
        for item_id, rating in ratings:
            feature.append((np.array(list(self.data.get_item_by_id(item_id).get_genres().values())), np.array(self.f(rating[0]))))

        return feature

    def create_training_examples_with_item_and_user_rating(self):
        """
        """
        raise NotImplementedError

    def training_and_test_for_an_user_with_item(self, user_id):
        """
        This method roughly does the following:

        1. find how many movies user_name_of(user_id) has rated
        2. divide the number of ratings to 80% (for training) and 20% (for test)
        3. create ndarray both form 80% and 20% of the information separately
        4. train it
        5. test it

        :param user_id: the id of the user for which you wanted to train and test
        :type user_id:  int
        """
        ratings_by_user_id = list(self.data.get_user_by_id(user_id).get_movie_rating().items())
        
        # find how many movies user_name_of(user_id) has rated
        nratings_by_user_id = len(ratings_by_user_id)

        # divide the number of ratings to 80% (for training) and 20% (for test)
        nratings_for_train  = int(nratings_by_user_id * .8)
        nratings_for_test   = nratings_by_user_id - nratings_for_train

        ratings_for_train = ratings_by_user_id[:nratings_for_train]
        ratings_for_test  = ratings_by_user_id[nratings_for_train:]

        # create ndarray both from 80% and 20% of the information separately
        train = self.create_training_examples_with_item(ratings_for_train)
        test  = self.create_training_examples_with_item(ratings_for_test)

        # train it
        NN = ann.Neural_Network()
        NN.backpropagation(train, 600, .5)

        for feature, y in test:
            print(convert.f_inverse_cap(list(NN.feedforward(feature)[0])), convert.f_inverse(list(y)))

        # test it

    def training_and_test_for_an_user_with_item_and_user_rating(self, user_id):
        """
        This method roughly does the following:

        1. sort neighbors according to the similarity
        2. find how many movies user_name_of(user_id) has rated
        3. divide the number of ratings to 80% (for training) and 20% (for test)
        4. create ndarray both form 80% and 20% of the information separately
        5. train it
        6. test it

        :param user_id: the id of the user for which you wanted to train and test
        :type user_id:  int
        """
        raise NotImplementedError
