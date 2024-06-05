from sklearn.svm import SVC 
import sklearn.metrics as metrics
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

test_est = clf.predict(X_test)
print('SVM精确度...')
print(metrics.classification_report(test_est, y_test))