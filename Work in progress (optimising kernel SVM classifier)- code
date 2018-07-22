""" Credit card fraud detection
Data from ULB machine learning group, credit card transactions made by european cardholders
over two days in september of 2013. There were 284,807 transactions of which 0.172% of total
transactions were classified as fraudulent. PCA was already performed on the original dataset
to preserve anonymity. In total there are 28 principal components and 30 input features in total."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset and overview
dataset = pd.read_csv('creditcard.csv')
dataset.columns
dataset.head(50)
dataset.tail(50)
dataset[['Time', 'Amount', 'Class']].describe()

# Analysing how many fraudulent classes there are
dataset['Class'].value_counts() """284315 non-fraud, 492 fraudulent"""
dataset['Class'].value_counts()/len(dataset) """roughly 99.8% non fraud, 0.173% fraud"""

# Plotting distribution
sns.factorplot(x = 'Class', kind = 'count', data = dataset) 
plt.title('Distribution of fraudulent and non-fraudulent transactions')
plt.xlabel('Class (Non-fraud & Fraud)')
plt.ylabel('Frequency')
plt.show()

# Describing fraud and non-fraud amounts
dataset_fraud = dataset[dataset['Class'] == 1]
dataset_nonfraud = dataset[dataset['Class'] ==0]
stats_fraud = dataset_fraud.describe()
stats_nonfraud = dataset_nonfraud.describe()

# Distribution plots of fraudulent and non-fraudulent transactions for each variable
import matplotlib.gridspec as gridspec
columns = dataset.iloc[:, 1:30].columns
frauds = dataset.Class == 1
non_frauds = dataset.Class == 0

gs = gridspec.GridSpec(14, 2)
plt.figure(figsize = (15, 100))

for n, col in enumerate(dataset[columns]):
    ax = plt.subplot(gs[n])
    sns.distplot(dataset[col][frauds], bins = 50, color = 'red')
    sns.distplot(dataset[col][non_frauds], bins = 50, color = 'blue')
    ax.set_ylabel('Density')
    ax.set_xlabel('')
    ax.set_title(str(col))
plt.show()

# Data preprocessing
"""Time and amount have to be scaled. PCA features were scaled before PCA was applied"""
from sklearn.preprocessing import StandardScaler
sc_time = StandardScaler()
sc_amount = StandardScaler()
dataset['Scaled time'] = sc_time.fit_transform(dataset['Time'].values.reshape(-1, 1))
dataset['Scaled amount'] = sc_amount.fit_transform(dataset['Amount'].values.reshape(-1, 1))

# Replacing Time and amount with scaled values
dataset.drop(['Time', 'Amount'], axis = 1, inplace = True)
scaled_t = dataset['Scaled time']
scaled_a = dataset['Scaled amount']
dataset.drop(['Scaled time', 'Scaled amount'], axis = 1, inplace = True)
dataset.insert(0, 'Scaled time', scaled_t)
dataset.insert(29, 'Scaled amount', scaled_a)

# Splitting dataset into training set and test set
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(test_size = 0.3, random_state = 0)
for train_index, test_index in split.split(dataset, dataset['Class']):
    train_set = dataset.loc[train_index]
    test_set = dataset.loc[test_index]
# Checking proportionality of stratified split
test_set['Class'].value_counts() / len(test_set) """Proportionality has been kept. 0.173% fraud cases"""
train_set['Class'].value_counts() / len(train_set) """proportionality has been kept. 0.173% fraud cases"""

"""In such a biased dataset which is heavily skewed, could possibly over sample 
the non-fraudulent transactions and over-sample the fraudulent transactions"""
# Split features from class
# Train set
train_set['Class'].value_counts()
test_set['Class'].value_counts()
y_train = train_set.iloc[:, 30].values
y_train['Class'].value_counts()
X_train = train_set.iloc[:, 0:30].values

# test set
X_test = test_set.iloc[:, 0:30].values
y_test = test_set.iloc[:, 30].values

# Fitting kernel SVM classifier to data
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'rbf', C = 1.0, random_state = 0)
svm_classifier.fit(X_train, y_train)

train_predict = svm_classifier.predict(X_train)

#Evaluating default model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, 
cm_train = confusion_matrix(y_train, train_predict)
precision_score(y_train, train_predict)
recall_score(y_train, train_predict)
f1_score(y_train, train_predict)
"""care more about recall as we don't want false negatives"""
from sklearn.model_selection import cross_val_predict
y_train_scores = cross_val_predict(svm_classifier, X_train, y_train, cv = 5, method = 'decision_function')

# Plot precision vs recall curve 
"""recall vs precision curves, good to use PR curve whenever the positive class is rare or if you care more
about FP than FN"""
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_scores)

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'b-', linewidth = 2)
    plt.xlabel('Recall', fontsize = 16)
    plt.ylabel('Precision', fontsize = 16)
    plt.axis([0, 1, 0, 1])
plt.figure(figsize = (8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()

from sklearn.model_selection import GridSearchCV
parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.5, 0.1, 0.001, 0.0001], 'kernel': ['rbf']},
        ]
grid_search = GridSearchCV(estimator = svm_classifier, param_grid = parameters, 
                           scoring = 'accuracy', cv = 3, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
grid_search.best_estimator_

new_svm_classifier = SVC()
new_svm_classifier.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = new_svm_classifier, X = X_train, y = y_train, cv = 3)
accuracies.mean()
accuracies.std() 

y_test_predict = new_svm_classifier.predict(X_test)

cm_test = confusion_matrix(y_test, y_test_predict)
precision_score(y_train, y_test_predict)
recall_score(y_train, y_test_predict)
f1_score(y_train, y_test_predict)


accuracy = TP + TN / TP + TN + FP + FN
