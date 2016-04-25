#!/usr/bin/env python
# -*- coding: utf-8 -*-

# library packages
import sys, pickle

# local packages
from mrs.recsys import predict

# usage: arg1: ['knn', 'ann', 'rbm']
#        arg2: user_id
#        arg3: item_id

if __name__ == '__main__':
    if sys.argv[1] == 'knn':
        with open('predicted_rating_knn.bin', 'rb') as f:
            rating = pickle.load(f)
            print(rating[int(sys.argv[2])][int(sys.argv[3])])
    elif sys.argv[1] == 'ann':
        with open('predicted_rating_ann.bin', 'rb') as f:
            rating = pickle.load(f)
            print(rating[int(sys.argv[2])][int(sys.argv[3])])
    elif sys.argv[1] == 'rbm':
        with open('predicted_rating_rbm.bin', 'rb') as f:
            rating = pickle.load(f)
            print(rating[int(sys.argv[2])][int(sys.argv[3])])
        #p = predict.PredictRBM()
        #p.load_hyperparameters()
        #print(p.predict(int(sys.argv[2]), int(sys.argv[3])))
