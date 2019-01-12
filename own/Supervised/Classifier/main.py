from sklearn.cross_validation import train_test_split, ShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, r2_score, fbeta_score, f1_score, accuracy_score, precision_score, recall_score
from own.Supervised.Classifier.visuals import ModelComplexity, ModelLearning, correlation
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np

# Reading data
data = pd.read_csv('processed.cleveland.data', sep=',', names=[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'disease'
])

# Replace columns names
# age
# sex male=1, female=0
# cp: chest pain type
# -- Value 1: typical angina
# -- Value 2: atypical angina
# -- Value 3: non-anginal pain
# -- Value 4: asymptomatic
# trestbps: resting blood pressure
# chol: serum cholestoral in mg/dl
# fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# restecg: resting electrocardiographic results
# -- Value 0: normal
# -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# thalach: maximum heart rate achieved
# exang: exercise induced angina (1 = yes; 0 = no)
# oldpeak = ST depression induced by exercise relative to rest
# slope: the slope of the peak exercise ST segment
# -- Value 1: upsloping
# -- Value 2: flat
# -- Value 3: downsloping
# ca: number of major vessels (0-3) colored by flourosopy
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# num: diagnosis of heart disease (angiographic disease status)
# -- Value 0: < 50% diameter narrowing
# -- Value 1: > 50% diameter narrowing

# Normalize disease feature
data.disease = data.disease.map(lambda x: x > 0 and 1 or 0)
data.ca = pd.to_numeric(data.ca, errors='coerce')
data.thal = pd.to_numeric(data.thal, errors='coerce')
data = data.dropna()

# Split data with numpy
X = np.array(data[data.columns[:-1]])
y = np.array(data[data.columns[-1:]])

# Split test and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# # LEARNING CURVE SCORE
# # Create 10 cross-validation sets for training and testing
# cv = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2, random_state=0)
# # # Generate the training set sizes increasing by 50
# train_sizes = np.rint(np.linspace(1, X.shape[0] * 0.8 - 1, 9)).astype(int)
#
# max_depth = np.arange(1, 30)
# # # Create three different models based on max_depth
# for k, depth in enumerate(max_depth):
#     # Create a Decision tree regressor at max_depth = depth
#     regressor = DecisionTreeClassifier(max_depth=depth)
#
#     # Calculate the training and testing scores
#     sizes, train_scores, test_scores = curves.learning_curve(
#         regressor, X, y, cv=cv, train_sizes=train_sizes,
#         scoring=make_scorer(fbeta_score, beta=0.5)
#     )
#
#     print('depth:', depth)
#     print('score train:', np.mean(train_scores))
#     print('score test:', np.mean(test_scores))


# # MODEL COMPLEX SCORE
# # Create 10 cross-validation sets for training and testing
# cv = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2, random_state=0)
# max_depth = np.arange(1, 10)
# # Create a Decision tree regressor at max_depth = depth
# regressor = DecisionTreeClassifier()
# # Calculate the training and testing scores
# train_scores, test_scores = curves.validation_curve(
#     regressor, X, y, cv=cv, param_name='max_depth', param_range=max_depth,
#     scoring=make_scorer(f1_score)
# )
#
# # For each depth
# for x, k in enumerate(train_scores):
#     print('depth', x + 1)
#     print('score train:', np.mean(train_scores[x]))
#     print('score test:', np.mean(test_scores[x]))


# GRID SEARCH
beta = 1
params = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'criterion': ['entropy', 'gini'],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}

# Scoring for regression
scoring = make_scorer(fbeta_score, beta=beta)
# Cross validation K-Fold
cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)

# Evaluate model
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=cv, scoring=scoring)
grid = grid.fit(X_train, y_train)
print(grid.best_params_)

regressor = grid.best_estimator_
y_predict = regressor.predict(X_test)

print(confusion_matrix(y_test, y_predict, labels=[1, 0]))

# Scores
print('F Score:', fbeta_score(y_test, y_predict, beta=beta))
print('Accuracy Score:', accuracy_score(y_test, y_predict))
print('Recall Score:', recall_score(y_test, y_predict))
print('Precision Score:', precision_score(y_test, y_predict))
