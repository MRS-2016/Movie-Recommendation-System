#!/usr/bin/env python
# -*- coding: utf-8 -*-

# library packages
from time import time
import pickle

# third party packages
import numpy as np

# local files
from ..datamodel import loaddata

class RBM_User:
    """
    Restricted Boltzmann Machine
    
    :param nvisible: Number of neurons on the visible layer
    :type nvisible:  int
    :param nhidden:  Number of neurons on the hidden layer
    :type nhidden:   int
    """
    def __init__(self, nvisible, nhidden):
        self._nvisible = nvisible
        self._nhidden  = nhidden

        # hyperparameters
        self.bvisible = np.random.randn(1, self._nvisible) # biases for visible layer
        self.bhidden  = np.random.randn(1, self._nhidden)  # biases for hidden  layer
        self.weights  = np.random.randn(self._nvisible, self._nhidden) # weights of the synapses

    def positive_phase(self, v):
        """
        Positive phase of the RBM.

        :param v: input to the visible layer
        :type v:  ndarray
        :return:  the output of the hidden layer
        :rtype:   ndarray
        """
        return self.logistic_function(v.dot(self.weights))

    def negative_phase(self, h):
        """
        Negative phase of the RBM

        This phase feeds input to the hidden layer and then propagates towards
        the visible layer and then again from the visible layer it propagates
        towards the hidden layer.

        :param h: input to the hidden layer from the opposite side
        :type h:  ndarray
        :return:  output of the visible layer when run through backwards and output of the hidden layer
        :rtype:   tuple
        """
        v = (h.dot(self.weights.T) + self.bvisible) / self._nhidden
        return v, self.positive_phase(v)

    def logistic_function(self, x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-self.bhidden + x))

class Trainer:
    def __init__(self):
        """
        class to train and test the RBM model
        """
        # load the data
        self.data = loaddata.Data()
        self.data.load_data()
        self.rating_matrix = self.data.get_rating_matrix_with_zero()

        # create the RBM
        self.rbm = RBM_User(self.rating_matrix.shape[1] - 1, 500)

    def split_rating_matrix_for_train_and_test(self):
        """
        The whole rating matrix is divided into 80% and 20% for training and testing respectively
        """
        ntrain = int(self.rating_matrix.shape[0] * .8)
        ntest  = self.rating_matrix.shape[0] - ntrain

        self.train_rating_matrix = self.rating_matrix[:ntrain, ]
        self.test_rating_matrix  = self.rating_matrix[ntrain:, ]

    def train(self, epoch, cdk, learning_rate):
        """
        Train the model

        :param epoch: Number of iterations
        :type epoch:  int
        :param cdk:   Number of iterations in negative phase of contrastive divergence
        :type cdk:    int
        :param learning_rate: learning rate of the model
        :type learning_rate:  float

        .. note:: The current state of the function doesn't consider the training and testing
        instead it considers the whole matrix as training. It prints the amount of time
        taken for each epoch and error at that point of time
        """

        self.split_rating_matrix_for_train_and_test()

        for _ in range(epoch):
            start = time()
            np.random.shuffle(self.rating_matrix)
            
            for v in self.rating_matrix:
                v_plus = np.array([v[1:]])
                # positive phase
                h_plus = self.rbm.positive_phase(v_plus)
                vh_plus = v_plus.T.dot(h_plus)

                # negative phase
                vh_minus = np.zeros(vh_plus.shape)
                v_minus  = np.zeros(v_plus.shape)
                h_minus  = np.zeros(h_plus.shape)
                hi = h_plus
                for cdi in range(cdk):
                    vi, hi = self.rbm.negative_phase(hi)
                    vh_minus += vi.T.dot(hi)
                    v_minus  += vi
                    h_minus  += hi
                vh_minus /= cdk
                v_minus  /= cdk
                h_minus  /= cdk

                # find out the error in weights
                delta_weights = learning_rate * (vh_plus - vh_minus)
                delta_v = learning_rate * (v_plus - v_minus)
                delta_h = learning_rate * (h_plus - h_minus)
                # make 0 to those rows of delta_weights whose visible neurons data is missing
                delta_weights[vh_plus == 0] = 0
                delta_v[v_plus == 0] = 0

                self.rbm.weights  += delta_weights
                self.rbm.bvisible += delta_v
                self.rbm.bhidden  += delta_h

            pickle.dump((self.rbm.bvisible, self.rbm.weights, self.rbm.bhidden), open('hyperparam.bin', 'wb'))
            print('time:', time() - start)
            error = 0
            n = 0
            for v in self.rating_matrix:
                v_plus = np.array([v[1:]])
                h_plus = self.rbm.positive_phase(v_plus)

                v_minus, h_minus = self.rbm.negative_phase(h_plus)

                error += np.sum((v_plus[v_plus != 0] - v_minus[v_plus != 0]) ** 2)
                n += np.count_nonzero(v_plus)

            print(_, (error / n) ** .5)
