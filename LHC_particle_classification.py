# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:45:38 2021

@author: jipoz
"""

import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
import time
from sklearn.metrics import roc_curve, roc_auc_score

# sys.stdout=open("external_file.txt","w")

start = time.time()

rows = 100000

print('\nRows used for model training: %.i' % rows)
train = pandas.read_csv('training.csv', nrows = rows)

label_class_correspondence = {'Electron': 0, 'Ghost': 1, 'Kaon': 2, 'Muon': 3, 'Pion': 4, 'Proton': 5}
class_label_correspondence = {0: 'Electron', 1: 'Ghost', 2: 'Kaon', 3: 'Muon', 4: 'Pion', 5: 'Proton'}

def get_class_ids(labels):
    return numpy.array([label_class_correspondence[alabel] for alabel in labels])

train['Class'] = get_class_ids(train.Label.values)

features = list(set(train.columns) - {'Label', 'Class'})

train, valid = train_test_split(train, random_state=42, train_size=0.80, test_size=0.20)

real_Class = valid.Class.values
# real_Class = train.Class.values


def AdaBoostClass(n_estimators, max_depth):
    
    t1 = time.time()

    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=0.01, random_state=42, base_estimator=DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=30, random_state=42))
    thisFit = clf.fit(train[features].values, train.Class.values)

    predict_Class = thisFit.predict(valid[features].values)
    prob_Class = clf.predict_proba(valid[features].values)

    accuracy = metrics.accuracy_score(real_Class, predict_Class)
    t2 = time.time()
    t = t2 - t1

    return (accuracy, t, predict_Class, prob_Class)

##########################################################################
##########################################################################
##########################################################################

accuracyDict_estimators = {}
timeDict_estimators = {}
accuracyDict_depth = {}
timeDict_depth = {}

print('\n#############   Analysis n_estimators   #############')

for i in range(20):
    a, t, predict_Class, prob_Class = AdaBoostClass((i+1)*10, 1)
    accuracyDict_estimators['{} & {}'.format((i+1)*10, 1)] = a
    timeDict_estimators['{} & {}'.format((i+1)*10, 1)] = t
        
plt.figure(figsize=(9,6))
plt.title('ACCURACY based on n_estimators',size=20)
plt.plot(range(10,210,10), accuracyDict_estimators.values())
plt.ylabel('Accuracy',size=15)
plt.xlabel('n_estimators',size=15)
plt.xticks(numpy.arange(10,210,10))
plt.xlim(10,200)
plt.show()

plt.figure(figsize=(9,6))
plt.title('EXECUTION TIME based on n_estimators',size=20)
plt.plot(range(10,210,10), timeDict_estimators.values())
plt.xlabel('n_estimators',size=15)
plt.ylabel('Exec time (s)',size=15)
plt.xticks(numpy.arange(10,210,10))
plt.xlim(10,200)
plt.show()

max_accuracy = max(accuracyDict_estimators, key=lambda key: accuracyDict_estimators[key])
print('Best n_estimators value: %s (default)' % max_accuracy)
print('MAX Accuracy: %.3f' % accuracyDict_estimators[max_accuracy])

##########################################################################
##########################################################################
##########################################################################

print('\n#############   Analysis max_depth   #############')

for i in range(16,62,2):
    a, t, predict_Class, prob_Class = AdaBoostClass(50, i)
    accuracyDict_depth['{} & {}'.format(50, i)] = a
    timeDict_depth['{} & {}'.format(50, i)] = t

plt.figure(figsize=(9,6))        
plt.title('ACCURACY based on max_depth',size=20)
plt.plot(range(16,62,2), accuracyDict_depth.values())
plt.ylabel('Accuracy',size=15)
plt.xlabel('max_depth',size=15)
plt.xticks(numpy.arange(16,62,2))
plt.xlim(16,60)
plt.show()

plt.figure(figsize=(9,6))
plt.title('EXECUTION TIME based on max_depth',size=20)
plt.plot(range(16,62,2), timeDict_depth.values())
plt.xlabel('max_depth',size=15)
plt.ylabel('Exec time (s)',size=15)
plt.xticks(numpy.arange(16,62,2))
plt.xlim(16,60)
plt.show()

max_accuracy = max(accuracyDict_depth, key=lambda key: accuracyDict_depth[key])
print('Best max_depth value: (default) %s' % max_accuracy)
print('MAX Accuracy: %.3f' % accuracyDict_depth[max_accuracy])

##########################################################################
##########################################################################
##########################################################################

# print('\n#############   Analysis n_estimators & max_depth (v.1)   #############')

# accuracyDict = {}
# timeDict = {}

# for i in range(20):
#     for j in range(20):
#         a, t, predict_Class, prob_Class = AdaBoostClass((j+1)*10, i+1)
#         accuracyDict['{} & {}'.format((j+1)*10, i+1)] = a
#         timeDict['{} & {}'.format((j+1)*10, i+1)] = t
      
# plt.figure(figsize=(9,6))
# plt.title('ACCURACY based on n_estimators & max_depth',size=20)
# plt.plot(range(1,401,1), accuracyDict.values())
# plt.ylabel('Accuracy',size=15)
# plt.xlabel('Number of posible combinations',size=15)
# plt.xticks(numpy.arange(1,401,49))
# plt.xlim(1,400)
# plt.ylim(0.6,1)
# plt.show()

# max_accuracy = max(accuracyDict, key=lambda key: accuracyDict[key])
# # print('MAX Accuracy combinations: %s' % max_accuracy)
# # print('MAX Accuracy: %.3f' % accuracyDict[max_accuracy])

# plt.figure(figsize=(9,6))
# plt.title('EXECUTION TIME based on n_estimators & max_depth',size=20)
# plt.plot(range(1,401,1), timeDict.values())
# plt.xlabel('Number of posible combinations',size=15)
# plt.ylabel('Exec time (s)',size=15)
# plt.xlim(1,400)
# plt.xticks(numpy.arange(1,401,49))
# plt.show()

# BEST_n_estimators = int(max_accuracy[0:3:1])
# BEST_max_depth = int(max_accuracy[5:8:1])

# print('BEST_n_estimators = %i' % BEST_n_estimators)
# print('BEST_max_depth = %i' % BEST_max_depth)
# print('\nMAX Accuracy: %.3f' % accuracyDict[max_accuracy])

##########################################################################
##########################################################################
##########################################################################

 print('\n#############   Analysis n_estimators & max_depth (v.2)   #############')

 param_grid = {"base_estimator__max_depth" : range(35,51),
               "n_estimators" :  range(50,110,10)
               }

ada = AdaBoostClassifier(learning_rate=0.01, base_estimator=DecisionTreeClassifier(min_samples_leaf=30))
gridSearch = GridSearchCV(ada, param_grid,n_jobs=-1,verbose=3)
gridSearch.fit(train[features].values, real_Class)

df = pandas.DataFrame(gridSearch.cv_results_)
print(df)
df.to_csv("results.csv")

##########################################################################
##########################################################################
##########################################################################

print('\n#############   Analysis best performance   #############')

BEST_n_estimators = 100
BEST_max_depth = 37

a, t, predict_Class, prob_Class = AdaBoostClass(BEST_n_estimators,BEST_max_depth)

unique1, counts1 = numpy.unique(real_Class, return_counts=True)
unique2, counts2 = numpy.unique(predict_Class, return_counts=True)
plt.figure(figsize=(9, 6))
plt.bar(unique1, counts1, width=0.4, label='Real Particle', color = 'blue')
plt.bar(unique2+0.42, counts2, width=0.4, label='Predicted Particle', color = 'green')
plt.title('Real vs Predicted particle detection', size=20)
plt.ylabel('Nº particles', size=15)
plt.legend(loc='lower right', fontsize=15)
plt.yticks(size=15)
plt.xticks(numpy.arange(6)+0.21,['Electron','Ghost','Kaon','Muon','Pion','Proton'], size=15)
plt.show()

print('\nACCURACY = %.3f' % a)

def ROC_curves(pred, labels):
    plt.figure(figsize=(9, 6))
    u_labels = numpy.unique(labels)
    for i in u_labels:
        y_true = labels == i
        y_pred = pred[:, i]
        FPR, TPR , thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        plt.plot(FPR, TPR, linewidth=3, label=class_label_correspondence[i] + ', AUC = ' + str(numpy.round(auc, 3)))
        plt.xlabel('False Positive Rate', size=15)
        plt.ylabel('True Positive Rate', size=15)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.legend(loc='lower right', fontsize=15)
        plt.title('ROC curves', size=20)
    plt.show()

ROC_curves(prob_Class, valid.Class.values)

##########################################################################
##########################################################################
##########################################################################

print('\n#############   Final Analysis   #############')

df = pandas.read_csv('results.csv')
df = df.iloc[: , 1:]

# Accuracy to mean fit time
plt.figure(figsize=(9,6))
plt.title('ACCURACY vs Mean Fit Time', size=20)
plt.plot(df["mean_fit_time"],df["mean_test_score"],linestyle = 'None',marker='o',markerfacecolor='b',markeredgecolor='b')
plt.ylabel('Accuracy', size=15)
plt.xlabel('Mean Fit Time', size=15)
plt.show()

# Accuracy to number of combinations
plt.figure(figsize=(9,6))
plt.title('ACCURACY', size=20)
plt.plot(df["mean_test_score"])
plt.ylabel('Accuracy', size=15)
plt.xlabel('Nº combinations', size=15)
plt.xlim(1,95)
plt.show()

##########################################################################
##########################################################################
##########################################################################

end = time.time()
time = (end - start)/60
print('\nTotal Time = %.2f mins' % time)
print('This is a log message')

# sys.stdout.close()






