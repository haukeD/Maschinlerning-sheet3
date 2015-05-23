"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machines classifier with
linear kernel.
"""
print __doc__

import numpy as np
import pylab as pl
from sklearn import svm

X = np.array([ [0,0], [0,2], [2,-1], [2,3] ])  # training points
Y = np.array([-1,1,-1,1])                      # corresponding labels

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w =  clf.coef_[0]
a = -w[0]/w[1]
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx = np.linspace(x_min, x_max)
yy = a*xx - (clf.intercept_[0])/w[1]


# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

# plot the line, the points, and the nearest vectors to the plane
pl.title('Linear SVM classification example')
pl.xlabel('x')
pl.ylabel('y')
pl.set_cmap(pl.cm.Paired)     # sets colormap
pl.plot(xx, yy, 'k-')         # 'k-': black solid line
pl.plot(xx, yy_down, 'k--')   # 'k--': black dashed line
pl.plot(xx, yy_up, 'k--')     # dito

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(X[:,0], X[:,1], c=Y)

pl.axis('tight')
pl.show()

