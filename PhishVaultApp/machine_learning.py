import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


# Step 2 read the csv files and create pandas dataframes
legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")


# Step 3 combine legitimate and phishing dataframes, and shuffle
df = pd.concat([legitimate_df, phishing_df], axis=0)

df = df.sample(frac=1)


# Step 4 remove'url' and remove duplicates, then we can create X and Y for the models, Supervised Learning
df = df.drop('URL', axis=1)

df = df.drop_duplicates()

X = df.drop('label', axis=1)
Y = df['label']


# Step 5 split data to train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)


# Step 6 create a ML model using sklearn
svm_model = svm.LinearSVC()

# Random Forest
rf_model = RandomForestClassifier(n_estimators=60)

# Decision Tree
dt_model = tree.DecisionTreeClassifier()

# KNeighborsClassifier
kn_model = KNeighborsClassifier()

# Step 7 train the model
svm_model.fit(x_train, y_train)


# Step 8 make some predictions using test data
predictions = svm_model.predict(x_test)


# Step 9 create a confusion matrix and tn, tp, fn , fp
tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()


# Step 10 calculate accuracy, precision and recall scores
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("accuracy --> ", accuracy)
print("precision --> ", precision)
print("recall --> ", recall)


# K-fold cross validation, and K = 5
K = 5
total = X.shape[0]
index = int(total / K)

# 1
X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[index:]

# 2
X_2_test = X.iloc[index:index*2]
X_2_train = X.iloc[np.r_[:index, index*2:]]
Y_2_test = Y.iloc[index:index*2]
Y_2_train = Y.iloc[np.r_[:index, index*2:]]

# 3
X_3_test = X.iloc[index*2:index*3]
X_3_train = X.iloc[np.r_[:index*2, index*3:]]
Y_3_test = Y.iloc[index*2:index*3]
Y_3_train = Y.iloc[np.r_[:index*2, index*3:]]

# 4
X_4_test = X.iloc[index*3:index*4]
X_4_train = X.iloc[np.r_[:index*3, index*4:]]
Y_4_test = Y.iloc[index*3:index*4]
Y_4_train = Y.iloc[np.r_[:index*3, index*4:]]

# 5
X_5_test = X.iloc[index*4:]
X_5_train = X.iloc[:index*4]
Y_5_test = Y.iloc[index*4:]
Y_5_train = Y.iloc[:index*4]


# X and Y train and test lists
X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]

Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test]


def calculate_measures(TN, TP, FN, FP):
    model_accuracy = (TP + TN) / (TP + TN + FN + FP)
    model_precision = TP / (TP + FP)
    model_recall = TP / (TP + FN)
    return model_accuracy, model_precision, model_recall


dt_accuracy_list, dt_precision_list, dt_recall_list = [], [], []
kn_accuracy_list, kn_precision_list, kn_recall_list = [], [], []


for i in range(0, K):
    # ----- DECISION TREE ----- #
    dt_model.fit(X_train_list[i], Y_train_list[i])
    dt_predictions = dt_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=dt_predictions).ravel()
    dt_accuracy, dt_precision, dt_recall = calculate_measures(tn, tp, fn, fp)
    dt_accuracy_list.append(dt_accuracy)
    dt_precision_list.append(dt_precision)
    dt_recall_list.append(dt_recall)

    # ----- K-NEIGHBOURS CLASSIFIER ----- #
    kn_model.fit(X_train_list[i], Y_train_list[i])
    kn_predictions = kn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=kn_predictions).ravel()
    kn_accuracy, kn_precision, kn_recall = calculate_measures(tn, tp, fn, fp)
    kn_accuracy_list.append(kn_accuracy)
    kn_precision_list.append(kn_precision)
    kn_recall_list.append(kn_recall)


DT_accuracy = sum(dt_accuracy_list) / len(dt_accuracy_list)
DT_precision = sum(dt_precision_list) / len(dt_precision_list)
DT_recall = sum(dt_recall_list) / len(dt_recall_list)

print("Decision Tree accuracy ==> ", DT_accuracy)
print("Decision Tree precision ==> ", DT_precision)
print("Decision Tree recall ==> ", DT_recall)

KN_accuracy = sum(kn_accuracy_list) / len(kn_accuracy_list)
KN_precision = sum(kn_precision_list) / len(kn_precision_list)
KN_recall = sum(kn_recall_list) / len(kn_recall_list)

print("K-Neighbours Classifier accuracy ==> ", KN_accuracy)
print("K-Neighbours Classifier precision ==> ", KN_precision)
print("K-Neighbours Classifier recall ==> ", KN_recall)
"""


"""

data = {'accuracy': [NB_accuracy, SVM_accuracy, DT_accuracy, RF_accuracy, AB_accuracy, NN_accuracy, KN_accuracy],
        'precision': [NB_precision, SVM_precision, DT_precision, RF_precision, AB_precision, NN_precision, KN_precision],
        'recall': [NB_recall, SVM_recall, DT_recall, RF_recall, AB_recall, NN_recall, KN_recall]
        }

index = ['NB', 'SVM', 'DT', 'RF', 'AB', 'NN', 'KN']

df_results = pd.DataFrame(data=data, index=index)


# visualize the dataframe
ax = df_results.plot.bar(rot=0)
plt.show()

































