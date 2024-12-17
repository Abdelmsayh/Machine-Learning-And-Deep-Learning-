from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_breast_cancer
import math
import time

data = load_breast_cancer()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = abs(pd.Series(data['target'])-1)  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Create a Linear regression model
ls = Ridge(alpha=0.01)
ls = LinearRegression()

# Fit the model to the training data
ls.fit(X_train, y_train)

# Calculate The Time
print(ls.score(X_test, y_test))

start_time = time.time()
end_time = time.time()
Time=end_time - start_time
print('Time: ', Time)

