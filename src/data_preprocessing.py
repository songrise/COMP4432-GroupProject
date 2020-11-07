# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

# %%
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

labels = train.columns.drop(['id', 'target'])
target = train['target']

# %%
TOP_FEATURES = 15

forest = ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=1)
forest.fit(train[labels], train['target'])

importances = forest.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in forest.estimators_],
    axis=0
)
indices = np.argsort(importances)[::-1]
indices = indices[:TOP_FEATURES]

print('Top features:')
for f in range(TOP_FEATURES):
    print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))

# %%
plt.figure()
plt.title('Top feature importances')
plt.bar(
    range(TOP_FEATURES),
    importances[indices],
    yerr=std[indices],
)
plt.xticks(range(TOP_FEATURES), indices)
plt.show()

# %%
