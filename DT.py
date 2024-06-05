import sklearn.tree as tree
import sklearn.metrics as metrics
from data import X_train, X_test, y_train, y_test

from sklearn.model_selection import GridSearchCV

param_grid = {'criterion': ['entropy', 'gini'],
             'max_depth': [2, 3, 4, 5, 6, 7, 8],
             'min_samples_split': [4, 8, 12, 16, 20, 24, 28]} 
            

clf = tree.DecisionTreeClassifier()  # 定义一棵树
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, 
                            scoring='roc_auc', cv=4) 

clfcv.fit(X=X_train, y=y_train)

# 使用模型来对测试集进行预测
test_est = clfcv.predict(X_test)

# 模型评估
import sklearn.metrics as metrics

print("决策树准确度:")
print(metrics.classification_report(y_test,test_est)) 