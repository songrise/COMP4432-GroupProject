#CAO Qun 19086364d
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn import metrics

def KNN(train,test):
    test = test.drop(['id'], axis=1)
    train_y = train['target']
    train_x = pd.DataFrame(train, columns=[str(i) for i in range(300)])
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(train_x,train_y)
    result = knn.predict_proba(test)
    ret = result[:,1]
    return ret

def LogR(train,test):
    test = test.drop(['id'], axis=1)
    train_y = train['target']
    train_x = pd.DataFrame(train, columns=[str(i) for i in range(300)])
    from sklearn.linear_model import LogisticRegression
    logreg =  LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
    logreg.fit(train_x,train_y)
    result = logreg.predict_proba(test)
    ret = result[:,1]
    return ret

def Tree(train, test):
    test = test.drop(['id'], axis=1)
    train_y = train['target']
    train_x = pd.DataFrame(train, columns=[str(i) for i in range(300)])
    from sklearn import tree
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(train_x,train_y)
    result = dtree.predict_proba(test)
    ret = result[:,1]
    import graphviz
    dot_data = tree.export_graphviz(dtree, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("dtree_pdf")
    return ret 

def RandomF(train, test):
    test = test.drop(['id'], axis=1)
    train_y = train['target']
    train_x = pd.DataFrame(train, columns=[str(i) for i in range(300)])
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(train_x,train_y)
    result = rf.predict_proba(test)
    ret = result[:,1]
    return ret 


test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')



#ret = KNN(train,test) #apply KNN
#ret = LogR(train, test) #apply LogR  
#ret = Tree(train, test) #apply Decision Tree
ret = RandomF(train, test) #apply Radom Forest
# get submission
sub = pd.read_csv('./data/sample_submission.csv')
sub['target'] = ret
sub.to_csv('./out/submission-'+str(time.time())+'.csv',index=False) 
