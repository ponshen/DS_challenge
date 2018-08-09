import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = 'datasets/Data.csv'

def data_processing():
	# Read input dataset and print the resulting dataframe 
	df = pd.read_csv(DATA_PATH)
	'''
	   Country   Age   Salary Purchased
	0   France  44.0  72000.0        No
	1    Spain  27.0  48000.0       Yes
	2  Germany  30.0  54000.0        No
	3    Spain  38.0  61000.0        No
	4  Germany  40.0      NaN       Yes
	5   France  35.0  58000.0       Yes
	6    Spain   NaN  52000.0        No
	7   France  48.0  79000.0       Yes
	8  Germany  50.0  83000.0        No
	9   France  37.0  67000.0       Yes
	'''


	# X: features, all columns without the last one
	# Y: values f(X), the last column
	# Use iloc[rows,cols] to select a subset of data in the dataframe
	# .values convert the dataframe into numpy.ndarray
	X = df.iloc[:, :-1].values
	Y = df.iloc[:, -1].values
	'''
	X = 
	[['France' 44.0 72000.0]
	 ['Spain' 27.0 48000.0]
	 ['Germany' 30.0 54000.0]
	 ['Spain' 38.0 61000.0]
	 ['Germany' 40.0 nan]
	 ['France' 35.0 58000.0]
	 ['Spain' nan 52000.0]
	 ['France' 48.0 79000.0]
	 ['Germany' 50.0 83000.0]
	 ['France' 37.0 67000.0]]

	 Y = ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
	'''

	# Use Imputer to fill the missing data with average values of other data
	imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
	imp = imp.fit(X[:, 1:3])
	X[:, 1:3] = imp.transform(X[:, 1:3])
	'''
	X = 
	[['France' 44.0 72000.0]
	 ['Spain' 27.0 48000.0]
	 ['Germany' 30.0 54000.0]
	 ['Spain' 38.0 61000.0]
	 ['Germany' 40.0 63777.77777777778]
	 ['France' 35.0 58000.0]
	 ['Spain' 38.77777777777778 52000.0]
	 ['France' 48.0 79000.0]
	 ['Germany' 50.0 83000.0]
	 ['France' 37.0 67000.0]]
	'''

	# Use LabelEncoder to encode strings into numbers
	# Encode the first feature into the one-of-K scheme
	labelencoder_X = LabelEncoder()
	X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
	'''
	X = 
	[[0 44.0 72000.0]
	 [2 27.0 48000.0]
	 [1 30.0 54000.0]
	 [2 38.0 61000.0]
	 [1 40.0 63777.77777777778]
	 [0 35.0 58000.0]
	 [2 38.77777777777778 52000.0]
	 [0 48.0 79000.0]
	 [1 50.0 83000.0]
	 [0 37.0 67000.0]]
	'''

	labelencoder_Y = LabelEncoder()
	Y = labelencoder_Y.fit_transform(Y)
	'''
	Y = [0 1 0 0 1 1 0 1 0 1]
	'''

	onehotencoder = OneHotEncoder(categorical_features=[0])		# encode col[0] into 1-to-k
	X = onehotencoder.fit_transform(X).toarray()
	'''
	col 0-2: Country (encoded)
	col 3: Age
	col 4: Salary
	X = 
	[[1.00000000e+00 0.00000000e+00 0.00000000e+00 4.40000000e+01 7.20000000e+04]
	 [0.00000000e+00 0.00000000e+00 1.00000000e+00 2.70000000e+01 4.80000000e+04]
	 [0.00000000e+00 1.00000000e+00 0.00000000e+00 3.00000000e+01 5.40000000e+04]
	 [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.80000000e+01 6.10000000e+04]
	 [0.00000000e+00 1.00000000e+00 0.00000000e+00 4.00000000e+01 6.37777778e+04]
	 [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.50000000e+01 5.80000000e+04]
	 [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.87777778e+01 5.20000000e+04]
	 [1.00000000e+00 0.00000000e+00 0.00000000e+00 4.80000000e+01 7.90000000e+04]
	 [0.00000000e+00 1.00000000e+00 0.00000000e+00 5.00000000e+01 8.30000000e+04]
	 [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.70000000e+01 6.70000000e+04]]
	'''

	# Splits the datasets into training sets and test sets
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
	'''
	X_train = 
	[[0.00000000e+00 1.00000000e+00 0.00000000e+00 4.00000000e+01 6.37777778e+04]
	 [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.70000000e+01 6.70000000e+04]
	 [0.00000000e+00 0.00000000e+00 1.00000000e+00 2.70000000e+01 4.80000000e+04]
	 [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.87777778e+01 5.20000000e+04]
	 [1.00000000e+00 0.00000000e+00 0.00000000e+00 4.80000000e+01 7.90000000e+04]
	 [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.80000000e+01 6.10000000e+04]
	 [1.00000000e+00 0.00000000e+00 0.00000000e+00 4.40000000e+01 7.20000000e+04]
	 [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.50000000e+01 5.80000000e+04]]

	Y_train = [1 1 1 0 1 0 0 1]

	X_test = 
	[[0.0e+00 1.0e+00 0.0e+00 3.0e+01 5.4e+04]
	 [0.0e+00 1.0e+00 0.0e+00 5.0e+01 8.3e+04]]

	Y_test = [0 0]
	'''

	# Feature Scaling x_i_fc = (x_i - avg(x_i)) / s_i
	sc_X = StandardScaler()
	X_train = sc_X.fit_transform(X_train)
	X_test = sc_X.fit_transform(X_test)
	'''
	X_train =
	[[-1.          2.64575131 -0.77459667  0.26306757  0.12381479]
	 [ 1.         -0.37796447 -0.77459667 -0.25350148  0.46175632]
	 [-1.         -0.37796447  1.29099445 -1.97539832 -1.53093341]
	 [-1.         -0.37796447  1.29099445  0.05261351 -1.11141978]
	 [ 1.         -0.37796447 -0.77459667  1.64058505  1.7202972 ]
	 [-1.         -0.37796447  1.29099445 -0.0813118  -0.16751412]
	 [ 1.         -0.37796447 -0.77459667  0.95182631  0.98614835]
	 [ 1.         -0.37796447 -0.77459667 -0.59788085 -0.48214934]]

	X_test =
	[[ 0.  0.  0. -1. -1.]
 	 [ 0.  0.  0.  1.  1.]]
	'''

	print("Input Data:")
	print(df)
	print("---------------")
	print("Processed Data:")
	print(np.hstack((X, Y[:, np.newaxis])))

data_processing()