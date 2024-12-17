from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
clf = make_pipeline(StandardScaler(), SVC(kernel='rbf',random_state=0,gamma='scale'))
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

start_time = time.time()
end_time = time.time()
Time=end_time - start_time
print('Time: ', Time)
