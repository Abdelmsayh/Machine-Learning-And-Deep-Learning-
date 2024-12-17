#K-Means Clustering
import time
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd

data = load_breast_cancer()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = abs(pd.Series(data['target'])-1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = KMeans(n_clusters=4)
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
Time=end_time - start_time
preds = model.predict(X_test.values) 
accuracy = metrics.accuracy_score(y_test, preds)
print("Time:",Time)
print("Accuracy: ", accuracy)