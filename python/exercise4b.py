"""
================================================
SVM: Maximum margin separating decision boundary
================================================

Apply nonlinear SVM to wine data set.
"""
print __doc__

import numpy as np
import pylab as pl
from sklearn import svm

X = np.loadtxt('wine_inputs.txt')
tmp = np.loadtxt('wine_targets.txt')
Y = np.array([[tmp[i]] for i in range(tmp.size)])
Y.shape = tmp.size


# split samples into training and test set
# training set: 2/3 of the samples, test set: 1/3 of the samples
numTestSamples = tmp.size/3 + 1
numTrainingSamples = tmp.size - numTestSamples
training = np.zeros(( numTrainingSamples, X.shape[1]))
target_training = np.zeros( numTrainingSamples )
test = np.zeros(( numTestSamples, X.shape[1]))
target_test = np.zeros( numTestSamples )
index_train = 0
index_test = 0
for i in range(tmp.size):
    if np.mod(i, 3) == 0:
        test[index_test] = X[i]
        target_test[index_test] = Y[i]
        index_test = index_test + 1
    else:
        training[index_train] = X[i]
        target_training[index_train] = Y[i]
        index_train = index_train + 1

print "input dimension: %d" % X.shape[1]
print "number of training samples: %d" % numTrainingSamples
print "number of test samples: %d" % numTestSamples
print ""

# fit the model
#clf = svm.SVC(kernel='poly', degree=2)
clf = svm.SVC(kernel='???') # FIX!!!
clf.fit(training, target_training)


# calculate training error 
training_error_svm = 0 # initialization
output_training = clf.predict(training) # classify training patterns 
for i in range(numTrainingSamples):
    if target_training[i] != output_training[i]:
        training_error_svm = training_error_svm + 1
print "number of misclassified training patterns: %d" % training_error_svm
training_error_svm = 1.0 * training_error_svm / numTrainingSamples        
print "training error SVM: %f" % training_error_svm
print ""

# calculate test error 
test_error_svm = 0 # initialization
output_test = clf.predict(test) # classify test patterns 
for i in range(numTestSamples):
    if target_test[i] != output_test[i]:
        test_error_svm = test_error_svm + 1
print "number of misclassified test patterns: %d" % test_error_svm
test_error_svm = 1.0 * test_error_svm / numTestSamples      
print "test error SVM: %f" % test_error_svm

