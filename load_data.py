"""
load your data, which should be a dictionary with
x_train, y_train, x_test, y_test Numpy arrays
defs files import data from here (from load_data import data)
"""

# this particular example loads data from a pickle file

import pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,load_breast_cancer

iris = load_breast_cancer()
x, y = iris.data, iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
data={ 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test }
# data_file = 'data/classification.pkl'

# print ("loading...")

# with open( data_file, 'rb' ) as f:
# 	data = pickle.load( f )

"""
data is a dict containing numpy arrays: 
{ 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test }
"""
