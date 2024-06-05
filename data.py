import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

data = np.loadtxt('data.txt')

X = data[:,0:5]
y = data[:,-1]


#标准化对象
#scaler = StandardScaler() # 标准化转换
#scaler.fit(X)  # 训练标准化对象
#X = scaler.transform(X)   # 转换数据集

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size=0.2, random_state=12345)

#交叉验证
#model = Classifier()
#scores = -1 * cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
#print("MAE scores:\n", scores)