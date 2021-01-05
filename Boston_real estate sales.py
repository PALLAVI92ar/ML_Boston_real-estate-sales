# Prediction of real estate price of a house in Boston city
# Rows-506, no of variables-15
# Uses Python, numpy, pandas, scikit-learn, matplotlib
# Predicting the sales of using linear regression and variable selection using correlation and individual coefficient test
# Making sure to not receive any multicollinearity issues
# Checking the assumptions  of normal distribution
# Worked on Gradient descent algorithm as well

# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing dataset
df = pd.read_csv("Boston.csv")

# Dataset dimensions 
df.shape

# Features data-type
df.info()

# List of features
list(df)

# Statistical summary
df.describe()

# Count of null values 
df.isnull().sum()

# Heatmap- finds correlation between Independent and dependent attributes
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), annot = True)
plt.show()

# split X and Y variables
X = df.iloc[:,1:14]
list(X)

Y = df['medv']
Y

# standardize the data
from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
X_scale = scaling.fit_transform(X)
X_scale.shape

# Split the train and test data sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size = 0.3 , random_state = 50) 
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# Launching the Linear Regresssion an fitting the model
from sklearn.linear_model import LinearRegression
m1 = LinearRegression()
m1.fit(X_train,Y_train)
m1.intercept_
m1.coef_

# predicting on training data
y_pred1 = m1.predict(X_train)

# Training Error
from sklearn.metrics import mean_squared_error
Training_error = mean_squared_error(Y_train,y_pred1)
print("Mse on Trained data",Training_error.round(4))

# predicting on testing data
y_pred2 = m1.predict(X_test)

# Test Error
from sklearn.metrics import mean_squared_error
Testing_error = mean_squared_error(Y_test,y_pred2)
print("Mse on Tested data",Testing_error.round(4))

# create two lissts for training error and test errors
training_error = []
test_error = []

random = range(1,10001)

# loop

for x in random:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size = 0.5 , random_state = x) 
    lm = LinearRegression()
    lm.fit(X_train,Y_train)
    
    y_pred1 = lm.predict(X_train)
    training_error.append(mean_squared_error(Y_train,y_pred1).round(4))
    
    y_pred2 = lm.predict(X_test)
    test_error.append(mean_squared_error(Y_test,y_pred2).round(4))

print(training_error)
print(test_error)

# visulalize the results
import matplotlib.pyplot as plt
plt.plot(random,training_error, label = "Training Errors")
plt.plot(random,test_error, label = "Test Errors")
plt.ylabel("Error rate")
plt.xlabel("Random")
plt.legend()
plt.show()

test_values = pd.DataFrame(test_error)
test_values.min()
test_values.max()


test_values.plot.hist()
test_values.describe()

##################################################################################
# SGD Regressor
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(learning_rate='constant', eta0=0.01)
sgd.fit(X_train,Y_train)
sgd.intercept_
sgd.coef_

Y_pred = sgd.predict(X_test)

import numpy as np
from sklearn.metrics import mean_squared_error
mse = (mean_squared_error(Y_test,Y_pred))
mse

alpha = np.arange(0.001, 0.01,0.001)
mse = []

for x in alpha:
    sgd = SGDRegressor(learning_rate='constant', eta0=x)
    sgd.fit(X_train,Y_train)
    Y_pred = sgd.predict(X_test)
    mse.append(mean_squared_error(Y_test,Y_pred))

print(mse)

err = pd.DataFrame(mse)

import matplotlib.pyplot as plt
plt.plot(alpha,err)
