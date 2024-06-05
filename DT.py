import sklearn.tree as tree

# 直接使用交叉网格搜索来优化决策树模型，边训练边优化
from sklearn.model_selection import GridSearchCV
# 网格搜索的参数：正常决策树建模中的参数 - 评估指标，树的深度，
 ## 最小拆分的叶子样本数与树的深度
param_grid = {'criterion': ['entropy', 'gini'],
             'max_depth': [2, 3, 4, 5, 6, 7, 8],
             'min_samples_split': [4, 8, 12, 16, 20, 24, 28]} 
                # 通常来说，十几层的树已经是比较深了

clf = tree.DecisionTreeClassifier()  # 定义一棵树
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, 
                            scoring='roc_auc', cv=4) 
        # 传入模型，网格搜索的参数，评估指标，cv交叉验证的次数
      ## 这里也只是定义，还没有开始训练模型

clfcv.fit(X=X_train, y=y_train)

# 使用模型来对测试集进行预测
test_est = clfcv.predict(X_test)

# 模型评估
import sklearn.metrics as metrics

print("决策树准确度:")
print(metrics.classification_report(y_test,test_est)) 