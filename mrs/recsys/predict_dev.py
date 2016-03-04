#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this module is responsible for loading the data by calling the necessary information from **loaddata** module from datamodel directory and then calling necessary algorithms to predict the score
# note that this will act as intermediary file for giving result by interacting with all other files

# third party packages
import numpy as np

# local files
from ..datamodel import loaddata, nitems, nusers
from . import cf
from . import ann_dev

class Predict:
    """
    It is responsible for predicting scores for an user to an item by using different techniques.

    Not matter what technique you use you will load the data first so it makes sense to load the data in __init__ method. Different techniques use different algorithms to calculate the prediction, so each method have its own separate implementation.
    """
    def __init__(self):
        self.data = loaddata.Data()
        self.data.load_data()
        
        self.rating_matrix      = self.data.get_rating_matrix()
        self.correlation_matrix = cf.Correlation().pearson(self.rating_matrix)

    def create_ndarray_for_features(self, user_id, _list, neighbors):
        """
        Rough algorithm::

            for every item_id in _list:
                initialize feature variable to empty list
                for neighbor_user in neighbors:
                    if neighbor_user has not rated item_id:
                        continue
                    if we got 40 ratings:
                        we are done. get out from the inner loop
                    add neighbor_user's rating to feature
                    initialize avg variable to average of values in feature variable
                    add avg to feature variable until we got 40 ratings
                    add the genre features to feature variable
        
        :param _list: contains item ids
        :type _list:  list
        """
        l = [0, 0, 0, 0, 0]
        all_feature = []
        y = []
        for item_id in _list:
            feature = []
            r = int(self.rating_matrix[user_id, item_id])
            y.append(l[:r - 1] + [1] + l[r:])
            #for neighbor_user, rating in neighbors:
                #if np.isnan(self.rating_matrix[neighbor_user, item_id]):
                    #continue
                #if len(feature) == 40:
                    #break
                #feature.append(self.rating_matrix[neighbor_user, item_id])

            #avg = sum(feature) / len(feature)
            #feature.extend([avg] * (40 - len(feature)))
            feature.extend(list(self.data.get_item_by_id(item_id).get_genres().values()))

            all_feature.append(feature)

        return np.array(all_feature), np.array([y])

    def sort_neighbors(self, user_id):
        """
        sort the neighbors in descending according to the correlation between users
        
        :param user_id: the id of the user for which the neighbors are going to be calculated
        :type user_id:  int
        """
        # sort in descending order by correlation value of np.nan filtered values from user ids with their correlation with user_id
        # l contains list of tuple of ids of users and correlation with user_name_of(user_id)
        l = sorted(list(filter(lambda v: -1. <= v[1] <= 1., zip(range(nusers + 1), self.correlation_matrix[user_id]))), key = lambda v: -v[1])
        return l

    def train_and_test_for_an_user(self, user_id):
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
        # train it
        # test it

        # sort neighbors according to the similarity
        neighbors = self.sort_neighbors(user_id)

        ratings_by_user_id = list(self.data.get_user_by_id(user_id).get_movie_rating())
        
        # find how many movies user_name_of(user_id) has rated
        nratings_by_user_id = len(ratings_by_user_id)

        # divide the number of ratings to 80% (for training) and 20% (for test)
        nratings_for_train  = int(nratings_by_user_id * .8)
        nratings_for_test   = nratings_by_user_id - nratings_for_train

        ratings_for_train = ratings_by_user_id[:nratings_for_train]
        ratings_for_test  = ratings_by_user_id[nratings_for_train:]

        # create ndarray both from 80% and 20% of the information separately
        train_feature, train_y = self.create_ndarray_for_features(user_id, ratings_for_train, neighbors)
        test_feature, test_y   = self.create_ndarray_for_features(user_id, ratings_for_test, neighbors)

        print(train_feature.shape, train_y.ravel().reshape((train_feature.shape[0], 1, 5)).shape)

        # train it
        NN = ann.Neural_Network()
        NN.backpropagation(train_feature, train_y.ravel().reshape((train_feature.shape[0], 1, 5)), .09)

        #print(NN.W[2])

        # test it
        for i in range(len(ratings_for_test)):
            print(self.scale(NN.feedforward(test_feature[i])), list(test_y[0][i]).index(1) + 1)

    @staticmethod
    def scale(l):
        l = l[0]
        no, val = 1, l[0]
        for i, x in enumerate(l):
            if x > val:
                no = i + 1
                val = x
        return no

    def predict_by_GD(self, user_id):
        # sort neighbors according to the similarity
        neighbors = self.sort_neighbors(user_id)

        ratings_by_user_id = list(self.data.get_user_by_id(user_id).get_movie_rating())
        
        # find how many movies user_name_of(user_id) has rated
        nratings_by_user_id = len(ratings_by_user_id)

        # divide the number of ratings to 80% (for training) and 20% (for test)
        nratings_for_train  = int(nratings_by_user_id * .8)
        nratings_for_test   = nratings_by_user_id - nratings_for_train

        ratings_for_train = ratings_by_user_id[:nratings_for_train]
        ratings_for_test  = ratings_by_user_id[nratings_for_train:]

        # create ndarray both from 80% and 20% of the information separately
        train_feature, train_y = self.create_ndarray_for_features(user_id, ratings_for_train, neighbors)
        test_feature, test_y   = self.create_ndarray_for_features(user_id, ratings_for_test, neighbors)

        training_example = []
        for X, y in list(zip(train_feature, train_y)):
            training_example.append((X, y))

        # train it
        NN = ann.Neural_Network()
        NN.GD(training_example, 0.003)

        #print(NN.W[2])

        # test it
        for i in range(len(ratings_for_test)):
            print(NN.feedforward(test_feature[i]), test_y[i])

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
        pass

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
        NN = ann_dev.Neural_Network()
        NN.backpropagation(train, 200, .003)

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
        pass
