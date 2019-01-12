import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split, ShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import sklearn.learning_curve as curves
from sklearn.metrics import make_scorer, r2_score, fbeta_score, f1_score, roc_auc_score, accuracy_score, \
    precision_score, recall_score
from own.Supervised.Classifier.visuals import ModelComplexity, ModelLearning, correlation, confusion_matrix_heatmap
from own.Supervised.Classifier.visuals2 import distribution, evaluate
from own.Supervised.Classifier.util import train_predict
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np

# Reading data
data = pd.read_csv('census.csv')
numerical_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categoric_features = [
    'workclass', 'education_level',
    'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
]

# Encoding categories to numeric
# Need know boolean
label_encoder = LabelEncoder()
data.income = label_encoder.fit_transform(data.income)

# Basic stats
# Check balancing for classification
# Is accuracy reliable?
# La descompensacion de los datos genera un mal calculo de la exactitud (accuracy)
# ex: 100TP + 1200TN / 100TP + 1200TN + 50FP + 50FN = 0.89

n = data.shape[0]
gt_than_50 = data[data.income > 0].shape[0]
lt_than_50 = data[data.income == 0].shape[0]
g_percent = (gt_than_50 / n) * 100

print("Ratio: >50:<50 =>", gt_than_50, ':', lt_than_50)
print(">50:", g_percent)
print("<50:", 100 - g_percent)

# Data
y = data.income
X = data.drop('income', axis=1)

# Pre processing categorical data
# Dummies encode
# One hot encoder
X = pd.get_dummies(X)

# Pre processing numerical Data
# Clean outliers
# distro before clean

# distribution(X[numerical_features])
X[numerical_features] = X[numerical_features].apply(
    lambda x: np.log(x + 1)
)

# Feature scaling
# Scaling numeric features to clean outliers
scaling = MinMaxScaler()
X[numerical_features] = scaling.fit_transform(X[numerical_features])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Calc time for estimators
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

clf_A = DecisionTreeClassifier(random_state=42)
clf_B = LogisticRegression()  # This doesnt accept random_state param
clf_C = GaussianNB()  # This doesnt accept random_state param
clf_D = RandomForestClassifier(random_state=42)
clf_E = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=4)
# clf_F = SVC(random_state=42, gamma=50)  # This doesnt accept random_state param

# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
beta = 0.5

# Collect results on the learners
for clf in [clf_A, clf_B, clf_C, clf_D, clf_E]:
    clf_name = clf.__class__.__name__
    result = train_predict(clf, samples_100, X_train, y_train, X_test, y_test, beta)
    print(
        '\nBeta:', beta,
        '\nTrain time:', result['train_time'],
        '\nPrediction time:', result['pred_time'],
        '\nAccuracy train:', result['acc_train'],
        '\nAccuracy test:', result['acc_test'],
        '\nF-Score train:', result['f_train'],
        '\nF-score test:', result['f_test']
    )

# distro transformed
# distribution(X[numerical_features], transformed=True)

params = {
    # Decision tree
    clf_A: {
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 5, 8, 10],
        'min_samples_leaf': [5, 6, 7],
        'max_features': [10, 20, 30, 40, 50],
        'random_state': [0, 42, 120]
    },
    # Logistic Regression
    clf_B: {
        'C': [0.1, 1.0, 1.5, 2.0],  # Alpha regularization (penalization for complexity) avoid overfit
        'random_state': [0, 42, 120]
    },
    # Naive Bayes
    clf_C: {},
    # RandomForest
    clf_D: {
        'n_estimators': [2, 4, 5, 6, 7],
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 5, 8, 10],
        'min_samples_leaf': [5, 6, 7],
        'random_state': [0, 42, 120]
    }
}

# scoring f-beta
score = make_scorer(fbeta_score, beta=0.5)
chosen_algorithm = clf_A
# K-cross Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
b_estimator = None

for estimator in [clf_A]:  # [clf_A, clf_B, clf_C]:
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=params[estimator],
        scoring=score, cv=cv
    )

    gs.fit(X_train, y_train)
    y_pred = gs.best_estimator_.predict(X_test)

    if estimator == clf_A:
        b_estimator = gs.best_estimator_

    print("\n" + estimator.__class__.__name__)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F-Score:', fbeta_score(y_test, y_pred, beta=beta))
    print('ROC Curve:', roc_auc_score(y_test, y_pred))
    # confusion_matrix_heatmap(y_test, y_pred)

# Boosting
boost = AdaBoostClassifier()
gs = GridSearchCV(
    estimator=boost,
    cv=cv, scoring=score,
    param_grid={
        'n_estimators': [2, 4, 6, 8, 10, 30],
        'base_estimator': [b_estimator]
    }
)

gs.fit(X_train, y_train)
y_pred = gs.best_estimator_.predict(X_test)

print("\n" + boost.__class__.__name__)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F-Score:', fbeta_score(y_test, y_pred, beta=beta))
print('F-Score Train:', fbeta_score(y_train, y_pred, beta=beta))
print('ROC Curve:', roc_auc_score(y_test, y_pred))
