#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas, os

class Data:
    def __init__(self):
        '''
        nusers: number of users in the movie dataset
        nitems: number of items in the movie dataset
        users:  list of user ids in the movie dataset
        items:  list of items in the movie dataset
        id_movie: it is a map from the movie ids to the name of the movies
        rating_matrix: It is a two dimensional matrix with users as columns and movies as rows
                       and each cell represents the rating given by the user for a particular movie
        '''
        self.nusers = None
        self.nitems = None
        self.users  = []
        self.items  = []
        self.id_movie = {}
        self.rating_matrix = None

    def n_user_item(self, path = None, encd = 'utf-8'):
        '''
        it finds out no of users and items currently present in the dataset
        param: path ==> The number of users, items and ratings in each line of the file.
        type: path ==> a file path
        param: encd ==> what kind of encoding is used by the file
        type: encd ==> string
        '''
        try:
            with open(path, encoding = encd) as f:
                d = f.readlines()
                self.nusers = int(d[0].split()[0])
                self.nitems = int(d[1].split()[0])
        except FileNotFoundError:
            print('The path provided to calculate no of users and items is not valid')
        except ValueError:
            print('You are casting a non integer object, strange!')

    def find_users(self):
        '''
        it specifies the user ids from known number of users, if the number of users is not known then it prints a message to stdout about that
        '''
        try:
            self.users = list(range(1, self.nusers + 1))
        except TypeError:
            print("Can't create user ids due to unknown number of users")

    def find_items(self):
        '''
        It creates ids for the movies depending on how much movie are there in the data set
        '''
        try:
            self.items = list(range(1, self.nitems + 1))
        except TypeError:
            print("Can't create item ids due to unknown number of users")

    def map_movie_ids(self, path = None, encd = 'utf-8'):
        '''
        it finds out what movies are associated with a specific id
        param: path ==> Each line of the file contains the tab separated list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
        The last 19 fields are the genres, a 1 indicates the movie
        is of that genre, a 0 indicates it is not; movies can be in
        several genres at once.
        type: path ==> a file path
        param: encd ==> what kind of encoding is used by the file
        type: encd ==> string
        '''
        try:
            for d in open(path, encoding = encd):
                _id, movie_name, *temp = d.split('|')
                self.id_movie[int(_id)] = movie_name
        except FileNotFoundError:
            print('The path provided to find out the movies is not valid')

    def build_user_item_matrix(self, path = None, encd = 'utf-8'):
        '''
        It build a matrix whose rows are represented by user ids and columns are represented as movie items.
        each cell represents the rating give by the user in that row to a prticular movie in that column.
        param: path ==> Each line of the file contains the tab separated list of
                             user id | item id | rating | timestamp.
        The time stamps are unix seconds since 1/1/1970 UTC.
        type: path ==> a file path
        param: encd ==> what kind of encoding is used by the file
        type: encd ==> string
        '''
        self.rating_matrix = pandas.DataFrame(index = self.items, columns = self.users, dtype = float)
        try:
            for d in open(path, encoding = encd):
                user_id, movie_id, rating, timestamp = map(int, d.split())
                self.rating_matrix[user_id][movie_id] = rating
        except FileNotFoundError:
            print('The path provided for the ratings by users is not valid')


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
