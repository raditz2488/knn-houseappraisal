import random
import sys

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Lets increase the recursionlimit as KDTree throws an error if it cant recurse 
sys.setrecursionlimit(10000)

class Regression(object):
    '''
    This class performs kNN regression
    '''

    def __init__(self):
        self.k = 5
        self.metric = np.mean
        self.kdtree = None
        self.houses = None
        self.prices = None

    def set_data(self, houses, prices):
        '''
            Creates a KDTree using houses
        '''
        self.houses = houses
        self.prices = prices
        self.kdtree = KDTree(self.houses)

    def regress(self, house):
        '''
        Predicts the price for the house.
        :param house: pandas.Series with house parameters
        :return: house price 
        '''
        # Get indexes of the k nearest neighbours of the house
        _, indexes = self.kdtree.query(house, self.k)

        # The indexes are in the prices indexing system. So it can be used to get the prices of the k nearest neighbours
        k_neighbours_prices = self.prices.iloc[indexes]

        # Find a value using the decided metric
        price = self.metric(k_neighbours_prices)

        # Return the price else throw an exception
        if np.isnan(price):
            raise Exception('Unexpected price')
        else:
            return price

class RegressionTest(object):
    '''
    Read the King's County housing data, calculate and plot the kNN regression error.
    '''

    def __init__(self):
        self.houses = None
        self.prices = None

    def load_csv_file(self, csv_file, limit=None):
        '''
        Loads CSV file with houses data
        :param csv_file: CSV file name
        :param limit: number of rows of file to read
        '''
        houses = pd.read_csv(csv_file, nrows=limit)
        # Filter prices column
        self.prices = houses['AppraisedValue']

        # Filter lat, long and SqFtLot columns
        house = houses[['lat', 'long', 'SqFtLot']]

        # Perform mean normalization
        # https://towardsdatascience.com/data-normalization-in-machine-learning-395fdec69d02
        houses = (houses - houses.mean()) / (houses.max() - houses.min())

        self.houses = houses




    