"""
================================================
SVM: Maximum margin separating decision boundary
================================================

Plot the maximum margin separating decision boundary within a two-class
separable dataset using a Support Vector Machines classifier.
"""
print __doc__

import numpy as np
import pylab as pl
from sklearn import svm

# load training data
X = np.loadtxt('svm_training_inputs.txt')         # training data
tmp = np.loadtxt('svm_training_targets.txt')      # training targets
Y = np.array([[tmp[i]] for i in range(tmp.size)])
Y.shape = tmp.size

# load test data
X_test = np.loadtxt('svm_test_inputs.txt') 


# create a mesh to plot in
h=.02 # step size in the mesh
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# fit the model  
clf = svm.SVC(kernel='???', ???) # FIX!!!
clf.fit(X, Y)


# classify test patterns
Y_test = clf.predict(X_test)
print "classification results for the test patterns:"
print Y_test


# plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# put the result into a color plot
Z = Z.reshape(xx.shape)
pl.set_cmap(pl.cm.Paired)     # sets colormap
pl.contourf(xx, yy, Z)
#pl.axis('off')

#plot also the training points
pl.scatter(X[:,0], X[:,1], c=Y)

# plot also the test points
pl.scatter(X_test[:,0], X_test[:,1], c=Y_test, s = 80)


pl.title('SVM data')
pl.xlabel('x1')
pl.ylabel('x2')
pl.axis('tight')
pl.show()

