from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from data import X_train, X_test, y_train, y_test

knn = KNeighborsClassifier(n_neighbors = 5) 
knn.fit(X_train, y_train)
test_est = knn.predict(X_test)

print("knn准确度:")
print(metrics.classification_report(y_test,test_est)) 