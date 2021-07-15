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
        # We dont need to normalize the prices as its not going to be used in distance calculations
        self.prices = houses['AppraisedValue']

        # Filter lat, long and SqFtLot columns
        house = houses[['lat', 'long', 'SqFtLot']]

        # Perform mean normalization because it helps in the distance calculations
        # https://towardsdatascience.com/data-normalization-in-machine-learning-395fdec69d02
        houses = (houses - houses.mean()) / (houses.max() - houses.min())

        self.houses = houses

    def test_regression(self, holdout):
        '''
        Prepares a train and test dataset from the current dataset and calculates regression for the test dataset.
        :param holdout part of the data to use as test data
        :return tuple(predicted_price, actual_price)
        '''
        # Filter the test and train dataset
        # Find the number of records in test dataset
        test_dataset_n = int(round(len(self.houses) * holdout))
        # Find indexes for the test dataset
        test_indexes = random.sample(self.houses.index, test_dataset_n)    

        # Find indexes for the train dataset
        train_indexes = set(self.houses.index) - set(test_indexes)

        # Prepare test dataset using indexes
        test_houses = self.houses.ix[test_indexes]

        # Prepare train dataset using indexes
        train_houses = self.houses.ix[train_indexes]
        train_prices = self.prices.ix[train_indexes]

        # Prepare Regression object on trian set
        regression = Regression()
        regression.set_data(train_houses, train_prices)

        # Prepare lists to capture predicted and actual values for the test set
        test_predicted_prices = []
        test_actual_prices = []

        # Predict values on test set using regression
        for idx, house in test_houses.iterrows():
            test_predicted_prices.append(regression.regress(house))
            test_actual_prices.append(self.prices.ix[idx])

        return test_predicted_prices, test_actual_prices

    def tests(self, folds):
        '''
        Calculates mean absolute errors for the folds
        :param folds: how may times the dataset should be split
        :return: list of error values
        '''
        
        # The proportion to use for test set
        holdout = 1 / folds
        errors = []
        for _ in range(folds):
            test_predicted_prices, test_actual_prices = self.test_regression(holdout)
            errors.append(mean_absolute_error(test_actual_prices, test_predicted_prices))

        return errors

    def plot_error_rates(self):
        folds_range = range(2, 11)
        errors_df = pd.DataFrame({'max': 0, 'min': 0}, index=folds_range)

        for folds in folds_range:
            errors = self.tests(folds)
            errors_df['max'][folds] = max(errors)
            errors_df['min'][folds] = min(errors)

        errors_df.plot(title='MAE of KNN over different number of folds')
        plt.xlabel('#folds')
        plt.ylabel('MAE')
        plt.show()

def main():
    regression_test = RegressionTest()
    regression_test.load_csv_file('king_county_data_geocoded.csv', 100)
    regression_test.plot




    