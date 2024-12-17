from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
Time=end_time - start_time
print("Time:",Time)
