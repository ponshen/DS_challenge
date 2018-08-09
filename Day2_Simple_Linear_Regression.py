import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

DATA_PATH = "datasets/studentscores.csv"

def simple_linear_regression():

	# Read input datasets
	df = pd.read_csv(DATA_PATH)
	'''
	    Hours  Scores
	0     2.5      21
	1     5.1      47
	2     3.2      27
	3     8.5      75
	4     3.5      30
	5     1.5      20
	6     9.2      88
	7     5.5      60
	8     8.3      81
	9     2.7      25
	10    7.7      85
	11    5.9      62
	12    4.5      41
	13    3.3      42
	14    1.1      17
	15    8.9      95
	16    2.5      30
	17    1.9      24
	18    6.1      67
	19    7.4      69
	20    2.7      30
	21    4.8      54
	22    3.8      35
	23    6.9      76
	24    7.8      86
	'''

	
	X = df.iloc[:, 0].values	# first column, convert to np.ndarray
	Y = df.iloc[:, 1].values	# second column, convert to np.ndarray
	'''
	# of datasets: 25
	X = [2.5 5.1 3.2 8.5 3.5 1.5 9.2 5.5 8.3 2.7 7.7 5.9 4.5 3.3 1.1 8.9 2.5 1.9 6.1 7.4 2.7 4.8 3.8 6.9 7.8]
	Y = [21 47 27 75 30 20 88 60 81 25 85 62 41 42 17 95 30 24 67 69 30 54 35 76 86]
	'''

	# No missing data, skip Imputer
	# Only number feature, skip encoding
	# Only one feature in X, no need to do feature scaling


	# Split datasets into train sets and test sets
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
	'''
	# of train sets: 18
	X_train = [7.8 6.9 1.1 5.1 7.7 3.3 8.3 9.2 6.1 3.5 2.7 5.5 2.7 8.5 2.5 4.8 8.9 4.5]
	Y_train = [86 76 17 47 85 42 81 88 67 30 25 60 30 75 21 54 95 41]
	
	# of test sets: 7
	X_test = [1.5 3.2 7.4 2.5 5.9 3.8 1.9]
	Y_test = [20 27 69 30 62 35 24]
	'''

	# Linear fitting on training sets
	# need to reshape X into 2D array 
	# reshape(-1, 1): reshape to 1 column with compatible row number
	regressor = LinearRegression()
	regressor = regressor.fit(X_train.reshape(-1,1), Y_train)
	

	# Predicting the result with the fitted regression
	Y_pred = regressor.predict(X_test.reshape(-1,1))
	'''
	Prediction values:
	Y_pred = [16.84472176 33.74557494 75.50062397 26.7864001  60.58810646 39.71058194 20.8213931 ]
	Real values:
	Y_test = [20 27 69 30 62 35 24]
	'''


	# Visualization
	plt.scatter(X_train, Y_train, color='red')	# training result
	plt.plot(X_train, regressor.predict(X_train.reshape(-1, 1)), color='blue')
	plt.show()

	plt.scatter(X_test, Y_test, color='red')  # test result
	plt.plot(X_test, Y_pred, color='blue')
	plt.show()


simple_linear_regression()