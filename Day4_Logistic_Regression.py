import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

DATA_PATH = 'datasets/Social_Network_Ads.csv'


def logistic_regression():
    # Importing input data
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, 2:-1].values
    Y = df.iloc[:, -1].values
    '''
          User ID  Gender  Age  EstimatedSalary  Purchased
    0    15624510    Male   19            19000          0
    1    15810944    Male   35            20000          0
    2    15668575  Female   26            43000          0
    3    15603246  Female   27            57000          0
    4    15804002    Male   19            76000          0
    5    15728773    Male   27            58000          0
    6    15598044  Female   27            84000          0
    7    15694829  Female   32           150000          1
    8    15600575    Male   25            33000          0
    9    15727311  Female   35            65000          0
    10   15570769  Female   26            80000          0
    11   15606274  Female   26            52000          0
    12   15746139    Male   20            86000          0
    13   15704987    Male   32            18000          0
    14   15628972    Male   18            82000          0
    15   15697686    Male   29            80000          0
    16   15733883    Male   47            25000          1
    17   15617482    Male   45            26000          1
    18   15704583    Male   46            28000          1
    19   15621083  Female   48            29000          1
    20   15649487    Male   45            22000          1
    21   15736760  Female   47            49000          1
    22   15714658    Male   48            41000          1
    23   15599081  Female   45            22000          1
    24   15705113    Male   46            23000          1
    25   15631159    Male   47            20000          1
    26   15792818    Male   49            28000          1
    27   15633531  Female   47            30000          1
    28   15744529    Male   29            43000          0
    29   15669656    Male   31            18000          0
    ..        ...     ...  ...              ...        ...
    370  15611430  Female   60            46000          1
    371  15774744    Male   60            83000          1
    372  15629885  Female   39            73000          0
    373  15708791    Male   59           130000          1
    374  15793890  Female   37            80000          0
    375  15646091  Female   46            32000          1
    376  15596984  Female   46            74000          0
    377  15800215  Female   42            53000          0
    378  15577806    Male   41            87000          1
    379  15749381  Female   58            23000          1
    380  15683758    Male   42            64000          0
    381  15670615    Male   48            33000          1
    382  15715622  Female   44           139000          1
    383  15707634    Male   49            28000          1
    384  15806901  Female   57            33000          1
    385  15775335    Male   56            60000          1
    386  15724150  Female   49            39000          1
    387  15627220    Male   39            71000          0
    388  15672330    Male   47            34000          1
    389  15668521  Female   48            35000          1
    390  15807837    Male   48            33000          1
    391  15592570    Male   47            23000          1
    392  15748589  Female   45            45000          1
    393  15635893    Male   60            42000          1
    394  15757632  Female   39            59000          0
    395  15691863  Female   46            41000          1
    396  15706071    Male   51            23000          1
    397  15654296  Female   50            20000          1
    398  15755018    Male   36            33000          0
    399  15594041  Female   49            36000          1
    '''
    
    # Splitting dataset into train set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train_norm = sc.fit_transform(X_train.astype(float))
    X_test_norm = sc.transform(X_test.astype(float))
    
    # Fitting Logistic Regression to the Training set
    classifier = LogisticRegression()
    classifier.fit(X_train_norm, Y_train)
    y_fitted = classifier.predict(X_train_norm)
    y_correct_train = (Y_train==y_fitted)   # True/False list
    #print(y_correct_train)

    #df_y_train = pd.DataFrame(data={'Y_train': Y_train, 'Y_fitted': y_fitted})
    #print(df_y_train)
    print("Fitted correctly: {:d}  Total: {:d}".format(np.sum(y_correct_train), y_correct_train.size))
    print("Fitting Accuracy: {:.1f} %".format(classifier.score(X_train_norm, Y_train) * 100))
    
    # Predicting Test set results
    y_pred = classifier.predict(X_test_norm)
    y_correct_test = (Y_test==y_pred)   # True False list
    #print(y_correct_test)

    #df_y_test = pd.DataFrame(data={'Y_test': Y_test, 'Y_pred': y_pred})
    #print(df_y_test)
    print("Predicted correctly: {:d}  Total: {:d}".format(np.sum(y_correct_test), y_correct_test.size))
    print("Prediction Accuracy: {:.1f} %".format(classifier.score(X_test_norm, Y_test) * 100))
    
    cm = confusion_matrix(Y_test, y_pred)
    print()
    print("True count in test y value: {:d}".format(np.sum(Y_test)))
    print("True count in pred y value: {:d}".format(np.sum(y_pred)))
    print()
    print("Confusion matrix:")
    print(cm)
    print()
    
    # Visualization
    # Age feature within [18, 60], Estimated Salary feature within [15000, 150000]
    # Plot the decision boundary where probability of y is 0.5
    X_true = X_test[np.where(Y_test==1)]
    X_false = X_test[np.where(Y_test==0)]
    xx, yy = np.mgrid[15:65:.1, 10000:160000:100]
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    probs = classifier.predict_proba(sc.transform(grid))[:,1].reshape(xx.shape)
    
    plt.contour(xx, yy, probs, [.5], cmap="Greys", vmin=0, vmax=.6)
    plt.scatter(X_true[:, 0], X_true[:, 1], c="b")
    plt.scatter(X_false[:, 0], X_false[:, 1], c="r")
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.show()


logistic_regression()