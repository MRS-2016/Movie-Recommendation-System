#!/usr/bin/env python
# -*- coding: utf-8 -*-

# create a class for correlation matrix and other class to find out similar users who have rated a particular movie w.r.t. a particular user based on correlation

import numpy as np
import pandas as pd

class Correlation:
    """
    """
    def pearson(self, rating_matrix):
        return pd.DataFrame(rating_matrix.T).corr().as_matrix()
