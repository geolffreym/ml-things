from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, make_scorer, roc_auc_score
from .visuals import ModelComplexity, ModelLearning, correlation, confusion_matrix_heatmap
from sklearn.model_selection import GridSearchCV, ShuffleSplit, learning_curve, validation_curve
from sklearn.feature_selection import RFE
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Generate dataframe
df = pd.read_csv('zoo.data', sep=',', names=[
    'animal', 'hair',
    'feather', 'eggs',
    'milk', 'airborne',
    'aquatic', 'predator',
    'toothed', 'backbone',
    'breathes', 'venomous',
    'fins', 'legs',
    'tail', 'domestic',
    'catsize', 'type'
])

# Split data
X = df.drop(['type', 'animal'], axis=1)
y = df['type']

# Preprocessing
# one_hot = OneHotEncoder()
# y = one_hot.fit_transform([y])

sns.heatmap(X.corr(), annot=True)
plt.show()

# Split test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Cross validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# # Generate C param
C = np.arange(1, 10)

# # LEARNING CURVE SCORE
# # Create three different models based on max_depth
for k, C in enumerate(C):
    # Create a Decision tree regressor at max_depth = depth
    regressor = LogisticRegression(C=C)

    # Calculate the training and testing scores
    sizes, train_scores, test_scores = learning_curve(
        regressor, X, y, cv=cv, n_jobs=4,
        scoring=make_scorer(accuracy_score)
    )

    print('C:', C)
    print('score train:', np.mean(train_scores))
    print('score test:', np.mean(test_scores))

# MODEL COMPLEX SCORE
# Calculate the training and testing scores
C = np.arange(1, 10)
regressor = LogisticRegression()
train_scores, test_scores = validation_curve(
    regressor, X, y, cv=cv, param_name='C', param_range=C,
    scoring=make_scorer(accuracy_score)
)

print('\n')
# For each depth
for x, k in enumerate(train_scores):
    print('C', x + 1)
    print('score train:', np.mean(train_scores[x]))
    print('score test:', np.mean(test_scores[x]))

# SUMMARY
# Best fit param C=2


# rfe = RFE(estimator, 10)
# rfe = rfe.fit(X, Y)
# print(rfe.support_)
# print(rfe.ranking_)

# Evaluate model
# Estimator
estimator = LogisticRegression()
grid = GridSearchCV(
    estimator, param_grid={'C': [1, 2, 4, 3, 5], 'penalty': ['l1', 'l2']},
    scoring='accuracy', cv=cv
)

grid.fit(X_train, y_train)
y_pred = grid.best_estimator_.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
# print('ROC:', roc_auc_score(y_test, y_pred, average='macro'))
# print('Accuracy:', precision_score(y_test, y_pred, average='weighted'))
