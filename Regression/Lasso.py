from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import time
import pandas as pd

data = load_breast_cancer()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = abs(pd.Series(data['target'])-1)  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Lasso regression model
ls = Lasso(alpha=0.01)

# Fit the model to the training data
ls.fit(X_train, y_train)

# Calculate The Time
print(ls.score(X_test, y_test))
start_time = time.time()
end_time = time.time()
Time=end_time - start_time
print('Time: ', Time)

