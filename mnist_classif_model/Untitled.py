
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_mldata
import numpy as np


# In[2]:


mnist = fetch_mldata('MNIST original')


# In[3]:


# Dependent and Independent var
# Input - output
# Features - Label
X, y = mnist['data'], mnist['target']


# In[4]:


# Basic stats
# Check balancing for classification
# Is accuracy reliable?
# La descompensacion de los datos genera un mal calculo de la exactitud (accuracy)
# ex: 100TP + 1200TN / 100TP + 1200TN + 50FP + 50FN = 0.89
from collections import Counter

n = X.shape[0]
count = Counter(y.tolist())

# Check proportions to lookup balance in classes 
for idx, val in dict(count).items():
    print("Proportion for class", int(idx), "=>", val / n)
    print("Count for class", int(idx), "=>", val)


# In[5]:


# get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt


# In[6]:


some_digit = X[36000]
some_digit = some_digit.reshape(28,28)
plt.imshow(some_digit, cmap=matplotlib.cm.binary)
plt.show()


# In[7]:


print(y[36000])


# In[8]:


from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, KFold, validation_curve, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[9]:


from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

def scorer(pred, classif): 
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    cross_val=np.mean(cross_val_score(classif, X_train, y_train, cv=cv, scoring='accuracy'))
    
    print('CrossVal Acc:', cross_val)
    print('Test Acc:', accuracy_score(y_test, pred), '\n')


def fit_predict(classif):
    import time
    before = time.time()
    print('Model:', classif.__class__.__name__)   
    classif.fit(X_train, y_train)
    train_time =  time.time() - before
    y_pred = classif.predict(X_train)
    y_pred = classif.predict(X_test)
    pred_time = time.time() - before
    scorer(y_pred, classif)
    val_time= time.time() - before
    
    print('Train Time:', train_time)
    print('Pred Time:', pred_time)
    print('Validation Time:', val_time, '\n', '\n')

    return classif
    
def fit_grid_predict(classif, params, scoring='accuracy'):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    import time
    before = time.time()
    
    print('Model:', classif.__class__.__name__)
    gs = GridSearchCV(classif, params, scoring, cv=cv)
    gs.fit(X_train, y_train)
    train_time =  time.time() - before
    
    y_pred = gs.predict(X_train)
    y_pred = gs.predict(X_test)
    pred_time = time.time() - before
    
    print('Train Time:', train_time)
    print('Pred Time:', pred_time)
    print('Best Score:', gs.best_score_, '\n', '\n')

    return gs.best_estimator_
    
def complex_curve(estimator, **kwargs):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    # MODEL COMPLEX SCORE
    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(
        estimator, X, y, cv=cv, **kwargs
    )
    
    print('\n')
    # For each depth
    for x, k in enumerate(train_scores):
        print('Param:', x + 1)
        print('score train:', np.mean(train_scores[x]))
        print('score test:', np.mean(test_scores[x]))


# In[10]:


from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import warnings
warnings.filterwarnings("ignore")

# Check param complex
# complex_curve(
#     DecisionTreeClassifier(), 
#     param_name='max_depth', 
#     param_range=np.arange(1,10),
#     scoring='accuracy'
# )

fit_predict(RandomForestClassifier(max_depth=5, n_estimators=10))
fit_predict(SGDClassifier(random_state=42))
fit_predict(DecisionTreeClassifier(max_depth=5))
fit_predict(LogisticRegression(multi_class='multinomial', solver='lbfgs'))
fit_predict(GaussianNB())



# In[14]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

# Tune params
forest = fit_grid_predict(RandomForestClassifier(), {'max_depth':[2,3,4,5], 'n_estimators':[5,8,10]})
# logit = fit_grid_predict(LogisticRegression(), {'multi_class':['multinomial'], 'solver':['lbfgs','newton-cg','saga']})                                                
sgd = fit_grid_predict(SGDClassifier(), {'loss':['hinge','log','perceptron'], 'learning_rate':['constant','optimal'], 'eta0':[0.2,0.5,0.8,0.9]})                                                







# In[15]:


from sklearn.externals import joblib
joblib.dump(forest, 'forest_classif.joblib') 
joblib.dump(sgd, 'sgd_classif.joblib') 


# In[17]:


from sklearn.metrics import confusion_matrix

#SGD estimator Choosen => best time training, best score
y_pred = sgd.predict(X_test)
con_matrix = confusion_matrix(y_test, y_pred)

plt.matshow(con_matrix, cmap=plt.cm.gray)
plt.show()

