from matplotlib import pyplot as plt
from sklearn import tree
from DT import clf

class_names = ['success', 'failure']
feature_names = ['Sodium alginate concentration','Calcium chloride concentration','Flow rate','Oil: Water','Shear rate']
fig = plt.figure(figsize=(15,15))
_ = tree.plot_tree(
    clf, 
    feature_names=feature_names,  
    filled=True,
    fontsize=10
)

# Save picture
fig.savefig("decistion_tree.png")