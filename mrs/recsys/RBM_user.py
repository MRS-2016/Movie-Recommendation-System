#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ..datamodel import loaddata

class RBM_User:
    def __init__(self, nvisible, nhidden):
        print(nvisible, nhidden)
        self._nvisible = nvisible
        self._nhidden  = nhidden

        # hyperparameters
        self.bvisible = np.random.randn(1, self._nvisible) # biases for visible layer
        self.bhidden  = np.random.randn(1, self._nhidden)  # biases for hidden  layer
        self.weights  = np.random.randn(self._nvisible, self._nhidden) # weights of the synapses


class Trainer:
    def __init__(self):
        # load the data
        self.data = loaddata.Data()
        self.data.load_data()
        self.rating_matrix = self.data.get_rating_matrix_with_zero()

        # create the RBM
        self.rbm = RBM_User(*self.rating_matrix.shape)

    def train(self):
        """
        train the model
        """

        
