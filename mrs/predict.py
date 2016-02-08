#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import Data
import pandas, os

'''
TODO: create every user in the data base. Then store the (sum of the ratings, number of ratings) given by each user
'''

class User:
    def __init__(self, user_vector):
        self.user_vector = user_vector

    def sort(self):
        '''
        It first removes all the **nan** values from the list and then sort in descending order.
        So that most similar user is at first index
        '''
        nan_eliminated = filter(lambda x: -1 <= x[1] <= 1, enumerate(self.user_vector))
        self.user_vector = sorted(map(lambda x: (x[0] + 1, x[1]), nan_eliminated), key = lambda x: x[1], reverse = True)

    def calculate_mean_rating(self):
        return sum(v[1] for v in self.user_vector) / len(self.user_vector)
        
class Predict:
    '''
    The current approach to calculate the prediction for a movie by a user is by using k-NN approach.
    It finds out the most similar n(=40) users for a particular user who also has rated that specific movie.
    Based on those users rating it predicts rating for the movie by that user.
    '''
    def __init__(self, data):
        self.data = data
        self.correlation_matrix = None

    # create a user for the user_id so further calculations for that user will be easier.
    def create_user(self, user_id):
        u = User(self.correlation_matrix[user_id])
        u.sort()
        return u

    def score(self, user_id, movie_id):
        '''
        Finds out the score that the name(user_id) will most likele give to the name(movie_id)
        $P_{a, i} = \overline{r_a} + \frac{\sum_{u=1}^{n} (r_{u, i} - \overline{r_u}) * w_{a, u}}{\sum_{u=1}^n w_{a, u}}$
        where $P_{a, i}$ is the prediction of person $a$ for item $i$
              $\overline{r_a}$ is the mean rating of user $a$
              $r_{u, i}$ is the rating given by user $a$ for item $i$
              $w_{a, u}$ is the weight between user $a$ and $u$ which makes some rating higher weight than others
              based on how similar the user $a$ is to user $u$
        '''
        a = self.create_user(user_id)
        mean_r_a = a.calculate_mean_rating()
        count, s, w, i = 40, 0, 0, -1
        
        while True:
            i += 1
            if i >= len(a.user_vector):
                break
            if count <= 0:
                break
            elif a.user_vector[i][0] == user_id:
                continue
            elif 1 <= self.data.rating_matrix[a.user_vector[i][0]][movie_id] <= 5:
                u = self.create_user(a.user_vector[i][0])
                mean_r_u = u.calculate_mean_rating()
                s += (self.data.rating_matrix[a.user_vector[i][0]][movie_id] - mean_r_u) * a.user_vector[i][1]
                w += a.user_vector[i][1]
                count -= 1
        res = mean_r_a + s / w
        #print(40 - count, mean_r_a, s , w)
        
        if 1 <= res <= 5:
            return res
        elif res < 1: return 1
        elif res > 5: return 5

    def create_correlation_matrix(self, mthd = 'pearson'):
        '''
        It creates the correlation matrix between every user to find out how closely they are related to each other in movie likings
        param: mthd ==> This tells which correlation algorithm should be used to find out the correlation matrix
                        It can be 'pearson' or 'kendall' or 'spearman'
        type:  mthd ==> string
        '''
        self.correlation_matrix = self.data.rating_matrix.corr(method = mthd)


if __name__ == '__main__':
    base = '/media/windows7/B.Tech/Project/ml-100k'
    
    d = Data()
    d.n_user_item(os.path.join(base, 'u.info'))
    print('No. of users: %d' % d.nusers)
    print('No. of items: %d' % d.nitems)
    d.find_users()
    d.find_items()
    d.map_movie_ids(os.path.join(base, 'u.item'), encd = 'CP1252')
    d.build_user_item_matrix(os.path.join(base, 'u.data'))
    print(d.rating_matrix)

    # for the prediction part
    p = Predict(d)
    p.create_correlation_matrix()
    print(p.correlation_matrix)
    from random import randint
    for i in range(10, 21):
        movie_id = i#randint(1, 1682)
        print('%s %s %s' % (d.id_movie[movie_id], p.score(10, movie_id), d.rating_matrix[10][movie_id]))
    '''
    for i in range(10, 31):
        u = User(p.correlation_matrix[i])
        u.sort()
        print('mean rating: %f' % u.calculate_mean_rating())
        print(p.score(i, 100))
    '''
