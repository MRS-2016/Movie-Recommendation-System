#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# local packages
from mrs.recsys.cf import Correlation

class KNN:
    '''
    It finds out the most similar n(=40) users for a particular user who also has rated that specific movie.
    Based on those users rating it predicts rating for the movie by that user.
    '''
    def __init__(self, data):
        self.data = data
        self.rating_matrix = data.get_rating_matrix_with_nan()
        self.correlation_matrix = Correlation().pearson(self.rating_matrix)

    def sort(self, user_id, user_vector):
        nan_eliminated = filter(lambda x: -1 <= x[1] <= 1, enumerate(self.user_vector))
        self.user_vector = sorted(nan_eliminated, key = lambda x: x[1], reverse = True)

    def score(self, user_id, movie_id):
        '''
        Finds out the score that the user will most likele give to the movie
        $P_{a, i} = \overline{r_a} + \frac{\sum_{u=1}^{n} (r_{u, i} - \overline{r_u}) * w_{a, u}}{\sum_{u=1}^n w_{a, u}}$
        where $P_{a, i}$ is the prediction of person $a$ for item $i$
              $\overline{r_a}$ is the mean rating of user $a$
              $r_{u, i}$ is the rating given by user $a$ for item $i$
              $w_{a, u}$ is the weight between user $a$ and $u$ which makes some rating higher weight than others
              based on how similar the user $a$ is to user $u$
        '''
        #print(self.rating_matrix[user_id][movie_id])
        mean_r_a = self.data.get_mean_rating_of_user(user_id)
        count, s, w = 40, 0, 0

        
        self.user_vector = self.correlation_matrix[user_id]
        self.sort(user_id, self.user_vector)

        for _user_id, w_a_u in self.user_vector:
            if count <= 0:
                break
            if _user_id == user_id:
                continue
            elif 1 <= self.rating_matrix[_user_id][movie_id] <= 5:
                mean_r_u = self.data.get_mean_rating_of_user(_user_id)
                s += (self.rating_matrix[_user_id][movie_id] - mean_r_u) * w_a_u
                w += w_a_u
                count -= 1
        
        try:
            res = mean_r_a + s / w
        except ZeroDivisionError:
            res = 1


        if 1 <= res <= 5:
            return res
        elif res < 1:
            return 1
        elif res > 5: return 5
