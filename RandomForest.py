from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from data import X_train, X_test, y_train, y_test

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

# 使用随机森林对测试集进行预测
test_est = rfc.predict(X_test)

print("knn准确度:")
print(metrics.classification_report(y_test,test_est)) 