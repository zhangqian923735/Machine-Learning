from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

log_model = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=1000)
log_model.fit(X_train,y_train)
test_est = log_model.predict(X_test)

print('逻辑回归精确度...')
print(metrics.classification_report(test_est, y_test))