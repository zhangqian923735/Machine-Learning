import numpy as np
data = np.loadtxt('data.txt')

X = data[:,0:5]
y = data[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size=0.2, random_state=12345)