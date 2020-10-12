# -*- coding : utf-8 -*-
# @FileName  : temp.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : 2020-10-12
# @Github    ：https://github.com/songrise
# @Descriptions: test

# %% import modules
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import numpy as np
# import seaborn as sns
import time

start = time.time()
# %% load data
train = np.loadtxt("..\\data\\train.csv", skiprows=1, delimiter=",")
test = np.loadtxt("..\\data\\test.csv", skiprows=1, delimiter=",")

# %% data process
train_label = train[:, 1]
train_data = train[:, 2:]
X_train, X_val, Y_train, Y_val = train_test_split(
    train_data, train_label)

# %% model setup
# clf = RandomForestClassifier(n_estimators=200, min_impurity_split=4, verbose=1)
clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(min_samples_split=4), random_state=int(time.time()))
clf.fit(X_train, Y_train)
clf.score(X_val, Y_val)
# save model
# joblib.dump(clf, "..\\model\\NNmodel_" +
# time.asctime(time.localtime(time.time())).replace(":", "_")+".csv")


# %% predict
result = clf.predict(test[:, 1:])
result = np.c_[np.arange(250, 20000).reshape(19750,), result]  # id column

np.savetxt("..\\out\\predictionNN_{}.csv".format(time.asctime(time.localtime(time.time())).replace(":", "_")), result, fmt="%.d",
           delimiter=",", header="id,target")

print("Done in {} sec.".format(time.time()-start))

# %%
