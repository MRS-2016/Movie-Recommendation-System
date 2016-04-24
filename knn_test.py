#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# library packages
import pickle

# third party packages
import numpy as np

# local files
from mrs.recsys.knn import KNN
from mrs.datamodel.loaddata import Data
from mrs.datamodel import nusers, nitems

if __name__ == '__main__':
    predicted_rating = np.zeros((nusers + 1, nitems + 1))
    data = Data()
    data.load_data()
    _knn = KNN(data)
    print(nusers, nitems)
    for u in range(1, nusers + 1):
        print('user with id', u, 'done')
        for i in range(1, nitems + 1):
            predicted_rating[u, i] = _knn.score(1, i)

    f = open('predicted_rating_knn.bin', 'wb')
    pickle.dump(predicted_rating, f)
