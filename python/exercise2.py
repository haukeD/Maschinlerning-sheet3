"""
=========================================
SVM and perceptron
=========================================

Artificially generate patterns,
artificially generate labels in a linearly separable fashion plus some noise,
generate a training and a test set in the same way,
train a SVM and a perceptron on the training set,
evaluate the classifiers on the test set.
"""
print __doc__

import numpy as np
import pylab as pl
import numpy.random as rand
from sklearn import svm
import neurolab as nl

###-----------------------------
# artificially generate patterns
###-----------------------------
 
# parameters
dim = 2                # dimension of input vector
center = np.array([[0.0, 0.0]]) # center 
true_weights = np.array([[ 1, 1]]) # true weight vector
sample_extension = 2.0 # parameter to determine the extension (spread) of the input patterns
sigma = 0.1            # noise parameter for generating the class labels

### test set
numTestSamples = ??? ### FIX!!! 
rand_norm = sample_extension * rand.randn(numTestSamples, 1, dim) # random values added to center 
test = np.array([center + r for r in rand_norm]) # generate test samples 
test.shape = (numTestSamples, dim)               # rearrange test array
# generate labels
target_test = np.array([0 for i in range(numTestSamples)])  # generate target for test
target_test.shape = ( numTestSamples, 1 )                   # rearrange test targets (for dot product)
for i in range(numTestSamples):
    target_test[i] = np.sign( np.dot(true_weights, test[i]) + sigma * rand.randn(1) )
target_test.shape = numTestSamples                          # rearrange test targets back

### training set
numTrainingSamples = ??? ### FIX!!!

rand_norm = sample_extension * rand.randn(numTrainingSamples, 1, dim) # random values added to center 
training = np.array([center + r for r in rand_norm]) # generate training samples 
training.shape = (numTrainingSamples, dim)        # rearrange training array
# generate labels
target_training = np.array([0 for i in range(numTrainingSamples)])  # generate target for training
target_training.shape = ( numTrainingSamples, 1 )                   # rearrange training targets (for dot product)
for i in range(numTrainingSamples):
    target_training[i] = np.sign( np.dot(true_weights, training[i]) + sigma * rand.randn(1) )
target_training.shape = numTrainingSamples                          # rearrange training targets back


###-----------------------------
# train SVM
###-----------------------------

clf = svm.SVC(kernel='linear')
clf.fit(training, target_training)

# get the separating hyperplane
w =  clf.coef_[0]
a = -w[0]/w[1]
x_min, x_max = training[:,0].min()-1, training[:,0].max()+1
y_min, y_max = training[:,1].min()-1, training[:,1].max()+1
xx = np.linspace(x_min, x_max)
yy = a*xx - (clf.intercept_[0])/w[1]
weights_svm_scaled = np.array([ 1, w[1]/w[0] ])
print "scaled weight vector SVM:"
print weights_svm_scaled

# calculate training error 
training_error_svm = 0 # initialization
output_training = clf.predict(training) # classify training patterns 
for i in range(numTrainingSamples):
    if target_training[i] != output_training[i]:
        training_error_svm = training_error_svm + 1
training_error_svm = 1.0 * training_error_svm / numTrainingSamples        
print "training error SVM: %f" % training_error_svm

# calculate test error 
test_error_svm = 0 # initialization
output_test = clf.predict(test) # classify test patterns 
for i in range(numTestSamples):
    if target_test[i] != output_test[i]:
        test_error_svm = test_error_svm + 1
test_error_svm = 1.0 * test_error_svm / numTestSamples      
print "test error SVM: %f" % test_error_svm

# plot training set with labels
pl.set_cmap(pl.cm.Paired)     # sets colormap
pl.scatter( training[:,0], training[:,1], c=target_training)

## # plot test set with labels
## pl.set_cmap(pl.cm.Paired)     # sets colormap
## pl.scatter( test[:,0], test[:,1], c=target_test, s=100)

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

# plot the line, the points, and the nearest vectors to the plane
pl.set_cmap(pl.cm.Paired)     # sets colormap
pl.plot(xx, yy, 'k-')         # 'k-': black solid line
pl.plot(xx, yy_down, 'k--')   # 'k--': black dashed line
pl.plot(xx, yy_up, 'k--')     # dito

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
print ""
print ""


###-----------------------------
# train perceptron
###-----------------------------

# create perceptron with 2 inputs and 1 output
net = nl.net.newff([[-1.0, 1.0],[-1.0, 1.0]], [1], transf=[nl.trans.TanSig()]) 
# Change train function
net.trainf = nl.train.train_gd
 
# initialise network 
#print "network initialisation:\n"
net.layers[0].np['w'][0] = [ 0, 0 ]
#print "weights after initialisation:" 
#print net.layers[0].np['w'][0]

# Bias output layer
net.layers[0].np['b'][0] = 0
#print "Bias after initialisation:"
#print net.layers[0].np['b'][0]

# rearrange array of labels
target_training.shape = ( numTrainingSamples, 1 )           
target_test.shape     = ( numTestSamples, 1 )               

### perceptron training
print "starting training\n"
numEpochs = 100
error = net.train(training, target_training, epochs=numEpochs, goal = 0.01, lr=0.1, show=10)
print "training finished\n"

print "weights after training:" 
w = net.layers[0].np['w'][0]
print w

print "Bias after training:"
b = net.layers[0].np['b'][0]
print b

# perceptron parameters:
w0 = b
w1 = w[0]
w2 = w[1]
if ( w2 == 0 ):
    print "Error: second weight zero!"
weights_perceptron_scaled = np.array([ 1, w[1]/w[0] ])
print "scaled weight vector perceptron:"
print weights_perceptron_scaled

# get the separating hyperplane
w =  clf.coef_[0]
a = -w1/w2
x_min, x_max = training[:,0].min()-1, training[:,0].max()+1
y_min, y_max = training[:,1].min()-1, training[:,1].max()+1
xx = np.linspace(x_min, x_max)
yy = a*xx - w0/w2
pl.plot(xx, yy, 'g-')         # 'g-': green solid line

# calculate training error 
training_error_perceptron = 0 # initialization
output_training = np.sign(net.sim(training)) # classify training patterns; sign function for binary labels
output_training.shape = numTrainingSamples # rearrange array
for i in range(numTrainingSamples):
    if target_training[i] != output_training[i]:
        training_error_perceptron = training_error_perceptron + 1
training_error_perceptron = 1.0 * training_error_perceptron / numTrainingSamples        
print "training error perceptron: %f" % training_error_perceptron

# calculate test error 
test_error_perceptron = 0 # initialization
output_test = np.sign(net.sim(test)) # classify test patterns; sign function for binary labels
output_test.shape = numTestSamples # rearrange array
for i in range(numTestSamples):
    if target_test[i] != output_test[i]:
        test_error_perceptron = test_error_perceptron + 1
test_error_perceptron = 1.0 * test_error_perceptron / numTestSamples      
print "test error perceptron: %f" % test_error_perceptron

print ""
print ""

###------------------------------------
# comparison of training and test error
###------------------------------------
print "comparison:"
print ""
print "number of training samples: %d" % numTrainingSamples
print ""
print "training error SVM: %f" % training_error_svm
print "test error SVM: %f" % test_error_svm
print "scaled weight vector SVM:"
print weights_svm_scaled
print ""
print "training error perceptron: %f" % training_error_perceptron
print "test error perceptron: %f" % test_error_perceptron
print "scaled weight vector perceptron:"
print weights_perceptron_scaled

pl.axis('tight')
pl.show()



