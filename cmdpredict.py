#!/usr/bin/env python
# -*- coding: utf-8 -*-

# library packages
import sys

# local packages
from mrs.recsys import predict

if __name__ == '__main__':
    p = predict.PredictRBM()
    p.load_hyperparameters()
    print(p.predict(sys.argv[1], sys.argv[2]))
