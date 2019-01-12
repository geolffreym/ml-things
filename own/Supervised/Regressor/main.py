from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, r2_score
from own.DecisionTreeRegressor.visuals import ModelComplexity, ModelLearning, correlation
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np

# Reading data
data = pd.read_csv('winequality_red.csv', sep=';')

# Replace columns names
data.columns = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_dioxide_sulfur', 'total_dioxide_sulfur',
    'density', 'ph', 'sulphates', 'alcohol', 'quality'
]

# fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";
# "free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"

# View correlation for data
correlation(data, data.columns)

data.drop('fixed_acidity', axis=1, inplace=True)
data.drop('residual_sugar', axis=1, inplace=True)
data.drop('chlorides', axis=1, inplace=True)
data.drop('free_dioxide_sulfur', axis=1, inplace=True)
data.drop('total_dioxide_sulfur', axis=1, inplace=True)
data.drop('density', axis=1, inplace=True)
data.drop('ph', axis=1, inplace=True)

# Split data with numpy
X = np.array(data[data.columns[:-1]])
y = np.array(data[data.columns[-1:]])

# Split test and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# View correlation for data
# correlation(data, ['volatile_acidity', 'citric_acid', 'sulphates', 'alcohol', 'quality'])
# ModelComplexity(X_train, y_train, np.arange(1, 10))
# ModelLearning(X, y)


# Scoring for regression
scoring = make_scorer(r2_score)
# Cross validation K-Fold
cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
# cv = KFold(n_splits=12)

# Decision Tree Regressor
regressor = DecisionTreeRegressor()
params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8], 'min_samples_split': [2, 3, 4, 5],
          'min_samples_leaf': [2, 3, 4, 5, 6, 7]}

# Evaluate model
grid = GridSearchCV(regressor, param_grid=params, cv=cv, scoring=scoring)
grid = grid.fit(X_train, y_train)

# Test model
y_predict = grid.best_estimator_.predict(X_test)
# Metrics for model
print("R2 Score:", r2_score(y_test, y_predict))

