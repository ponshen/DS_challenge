import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

DATA_PATH = "datasets/50_Startups.csv"

def multi_regression():
    
    # Import the datasets
    # Read the input into a dataframe
    # Get ndarray X for features and Y for values
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    '''
        R&D Spend  Administration  Marketing Spend       State     Profit
    0   165349.20       136897.80        471784.10    New York  192261.83
    1   162597.70       151377.59        443898.53  California  191792.06
    2   153441.51       101145.55        407934.54     Florida  191050.39
    3   144372.41       118671.85        383199.62    New York  182901.99
    4   142107.34        91391.77        366168.42     Florida  166187.94
    5   131876.90        99814.71        362861.36    New York  156991.12
    6   134615.46       147198.87        127716.82  California  156122.51
    7   130298.13       145530.06        323876.68     Florida  155752.60
    8   120542.52       148718.95        311613.29    New York  152211.77
    9   123334.88       108679.17        304981.62  California  149759.96
    10  101913.08       110594.11        229160.95     Florida  146121.95
    11  100671.96        91790.61        249744.55  California  144259.40
    12   93863.75       127320.38        249839.44     Florida  141585.52
    13   91992.39       135495.07        252664.93  California  134307.35
    14  119943.24       156547.42        256512.92     Florida  132602.65
    15  114523.61       122616.84        261776.23    New York  129917.04
    16   78013.11       121597.55        264346.06  California  126992.93
    17   94657.16       145077.58        282574.31    New York  125370.37
    18   91749.16       114175.79        294919.57     Florida  124266.90
    19   86419.70       153514.11             0.00    New York  122776.86
    20   76253.86       113867.30        298664.47  California  118474.03
    21   78389.47       153773.43        299737.29    New York  111313.02
    22   73994.56       122782.75        303319.26     Florida  110352.25
    23   67532.53       105751.03        304768.73     Florida  108733.99
    24   77044.01        99281.34        140574.81    New York  108552.04
    25   64664.71       139553.16        137962.62  California  107404.34
    26   75328.87       144135.98        134050.07     Florida  105733.54
    27   72107.60       127864.55        353183.81    New York  105008.31
    28   66051.52       182645.56        118148.20     Florida  103282.38
    29   65605.48       153032.06        107138.38    New York  101004.64
    30   61994.48       115641.28         91131.24     Florida   99937.59
    31   61136.38       152701.92         88218.23    New York   97483.56
    32   63408.86       129219.61         46085.25  California   97427.84
    33   55493.95       103057.49        214634.81     Florida   96778.92
    34   46426.07       157693.92        210797.67  California   96712.80
    35   46014.02        85047.44        205517.64    New York   96479.51
    36   28663.76       127056.21        201126.82     Florida   90708.19
    37   44069.95        51283.14        197029.42  California   89949.14
    38   20229.59        65947.93        185265.10    New York   81229.06
    39   38558.51        82982.09        174999.30  California   81005.76
    40   28754.33       118546.05        172795.67  California   78239.91
    41   27892.92        84710.77        164470.71     Florida   77798.83
    42   23640.93        96189.63        148001.11  California   71498.49
    43   15505.73       127382.30         35534.17    New York   69758.98
    44   22177.74       154806.14         28334.72  California   65200.33
    45    1000.23       124153.04          1903.93    New York   64926.08
    46    1315.46       115816.21        297114.46     Florida   49490.75
    47       0.00       135426.92             0.00  California   42559.73
    48     542.05        51743.15             0.00    New York   35673.41
    49       0.00       116983.80         45173.06  California   14681.40
    '''
    
    
    # Encoding Categorical Data
    # Encode the "State" column data from string to number
    # Dummy variable will occupy first columns followed by unencoded features
    labelencoder = LabelEncoder()
    X[:, 3] = labelencoder.fit_transform(X[:, 3])
    onehotencoder = OneHotEncoder(categorical_features = [3])
    X = onehotencoder.fit_transform(X).toarray()
    
    
    # Avoiding Dummy Variable Trap
    # Get rid of first column (must be one of dummy variables)
    X = X[:, 1:]
    
    
    # Spliting the dataset into training set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    
    # Fitting Multiple Linear Regression to the Training set
    # normalize=True: feature normalization of X before regression
    regressor = LinearRegression(normalize=True)
    regressor.fit(X_train, Y_train)
    train_score = regressor.score(X_train, Y_train)
    print("Result of linear regression on train set...")
    df_train_y = pd.DataFrame(data={'Fitted y': np.around(regressor.predict(X_train)), 'Real y': np.around(Y_train)})
    print(df_train_y)
    print("train score: {:.2f}".format(train_score))
    print()

    
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    test_score = regressor.score(X_test, Y_test)
    print("Predicting y on test set...")
    df_test_y = pd.DataFrame(data={'Pred. y': np.around(y_pred), 'Real y': np.around(Y_test)})
    print(df_test_y)
    print("test score: {:.2f}".format(test_score))


    
    
multi_regression()