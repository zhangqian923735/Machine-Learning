from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics
from data import X_train, X_test, y_train, y_test

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_predict = mnb.predict(X_test)

print(metrics.classification_report(y_test, y_predict)) 