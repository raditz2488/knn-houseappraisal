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
    