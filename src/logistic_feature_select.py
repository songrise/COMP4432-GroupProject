import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import euclidean_distances, make_scorer, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.regression import r2_score

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


test = pd.read_csv('..\\data\\test.csv')
test = test.drop(['id'], axis=1)
train = pd.read_csv('..\\data\\train.csv')


# LogisticRegression
# rec = []
# for i in range(300):
#     rec.append((abs(train['target'].corr(train[str(i)])),i))
# # print(sorted(rec,reverse=True))
# save = [i[1] for i in sorted(rec,reverse=True)]
train_y = train['target']
train_x = pd.DataFrame(train, columns=[str(i) for i in range(300)])
train_x = train_x.values
test = test.values
# x = train.drop(["id","target"],axis=1)
model =  LogisticRegression(solver='liblinear', class_weight='balanced', C=0.25, penalty='l1')
# model =  LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', C=0.31, penalty='l1')
param_grid = {
        'C'     : [i/100 for i in range(10,30,1)],
        'tol'   : [0.0001, 0.00011, 0.00009]
    }

def scoring_roc_auc(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5

feature_selector = RFECV(model, min_features_to_select=15, scoring=make_scorer(scoring_roc_auc), verbose=0, cv=20, n_jobs=-1)


print("counter | val_mse  |  val_mae  |  val_roc  |  val_cos  |  val_dist  |  val_r2    | best_score | feature_count ")
print("-------------------------------------------------------------------------------------------------")

predictions = pd.DataFrame()
counter = 0
cnt = 0
# split training data to build one model on each traing-data-subset
for train_index, val_index in StratifiedShuffleSplit(n_splits=1000, test_size=0.35).split(train_x, train_y):
    X, val_X = train_x[train_index], train_x[val_index]
    y, val_y = train_y[train_index], train_y[val_index]

    # get the best features for this data set
    feature_selector.fit(X, y)

    # remove irrelevant features from X, val_X and test
    X_important_features        = feature_selector.transform(X)
    val_X_important_features    = feature_selector.transform(val_X)
    test_important_features     = feature_selector.transform(test)

    # run grid search to find the best Lasso parameters for this subset of training data and subset of features 
    grid_search = GridSearchCV(feature_selector.estimator_, param_grid=param_grid, verbose=0, n_jobs=-1, scoring=make_scorer(roc_auc_score), cv=20)
    grid_search.fit(X_important_features, y)
    y_pred = grid_search.best_estimator_.predict_proba(X_important_features)[:,1]

    # score our fitted model on validation data
    val_y_pred = grid_search.best_estimator_.predict_proba(val_X_important_features)[:,1]
    val_mse = mean_squared_error(val_y, val_y_pred)
    val_mae = mean_absolute_error(val_y, val_y_pred)
    val_roc = roc_auc_score(val_y, val_y_pred)
    val_cos = cosine_similarity(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_dst = euclidean_distances(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_r2  = r2_score(val_y, val_y_pred)

    # if model did well on validation, save its prediction on test data, using only important features
    # r2_threshold (0.185) is a heuristic threshold for r2 error
    # you can use any other metric/metric combination that works for you
    if val_r2 > 0.185:
        message = '<-- OK'
        prediction = grid_search.best_estimator_.predict_proba(test_important_features)[:,1]
        predictions = pd.concat([predictions, pd.DataFrame(prediction)], axis=1)
        cnt+=1
        if cnt > 10:
            break
    else:
        message = '<-- skipping'


    print("{:2}      | {:.4f}   |  {:.4f}   |  {:.4f}   |  {:.4f}   |  {:.4f}    |  {:.4f}    |  {:.4f}    |  {:3}         {}  ".format(counter, val_mse, val_mae, val_roc, val_cos, val_dst, val_r2, grid_search.best_score_, feature_selector.n_features_, message))
    
    counter += 1



# get submission
# ret = model.predict(test.drop("id",axis=1))
ret = predictions.mean(axis=1)
sub = pd.read_csv('..\\data\\sample_submission.csv')
sub['target'] = ret
# sub['target'] = sub['target'].astype(int)
sub.to_csv('..\\out\\submission-'+str(time.time())+'.csv',index=False) 
