from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x, Y)
print(x.shape)

# 使用随机森林对测试集进行预测
test_est = rfc.predict(X_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))